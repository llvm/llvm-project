#!/usr/bin/env python3
"""
Resilience Utilities - Retry Logic & Graceful Degradation

Production-grade retry logic, circuit breakers, and fallback mechanisms
for Screenshot Intelligence System

Features:
- Exponential backoff retry with jitter
- Circuit breaker pattern
- Graceful degradation
- Timeout handling
- Error rate monitoring
- Automatic fallback to backup systems

Usage:
    from resilience_utils import with_retry, CircuitBreaker, with_timeout

    @with_retry(max_attempts=3, backoff_factor=2.0)
    def risky_operation():
        ...

    circuit_breaker = CircuitBreaker(failure_threshold=5, timeout=60)
    result = circuit_breaker.call(risky_function, arg1, arg2)
"""

import time
import logging
import functools
from typing import Callable, Any, Optional, Type, Tuple
from datetime import datetime, timedelta
import random

logger = logging.getLogger(__name__)


class RetryExhausted(Exception):
    """Raised when all retry attempts are exhausted"""
    pass


class CircuitBreakerOpen(Exception):
    """Raised when circuit breaker is open"""
    pass


def with_retry(
    max_attempts: int = 3,
    backoff_factor: float = 2.0,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    jitter: bool = True,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    on_retry: Optional[Callable] = None
):
    """
    Decorator for retry logic with exponential backoff

    Args:
        max_attempts: Maximum number of attempts
        backoff_factor: Exponential backoff multiplier
        initial_delay: Initial delay in seconds
        max_delay: Maximum delay in seconds
        jitter: Add random jitter to prevent thundering herd
        exceptions: Tuple of exceptions to catch and retry
        on_retry: Callback function called on each retry

    Example:
        @with_retry(max_attempts=5, backoff_factor=2.0)
        def fetch_data():
            return requests.get(url)
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay
            last_exception = None

            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e

                    if attempt == max_attempts:
                        logger.error(
                            f"{func.__name__} failed after {max_attempts} attempts: {e}"
                        )
                        raise RetryExhausted(
                            f"Failed after {max_attempts} attempts"
                        ) from e

                    # Calculate delay with exponential backoff
                    current_delay = min(delay, max_delay)
                    if jitter:
                        current_delay *= (0.5 + random.random())  # Add 0-50% jitter

                    logger.warning(
                        f"{func.__name__} attempt {attempt}/{max_attempts} failed: {e}. "
                        f"Retrying in {current_delay:.2f}s..."
                    )

                    if on_retry:
                        on_retry(attempt, e, current_delay)

                    time.sleep(current_delay)
                    delay *= backoff_factor

            raise last_exception

        return wrapper
    return decorator


def with_timeout(seconds: float):
    """
    Decorator to add timeout to functions

    Args:
        seconds: Timeout in seconds

    Example:
        @with_timeout(30.0)
        def slow_operation():
            ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            import signal

            def timeout_handler(signum, frame):
                raise TimeoutError(f"{func.__name__} timed out after {seconds}s")

            # Set alarm
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(int(seconds))

            try:
                result = func(*args, **kwargs)
            finally:
                # Cancel alarm
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)

            return result

        return wrapper
    return decorator


class CircuitBreaker:
    """
    Circuit breaker pattern implementation

    States:
    - CLOSED: Normal operation
    - OPEN: Failures exceeded threshold, reject calls
    - HALF_OPEN: Testing if service recovered

    Usage:
        breaker = CircuitBreaker(failure_threshold=5, timeout=60)
        result = breaker.call(risky_function, arg1, arg2)
    """

    STATE_CLOSED = 'CLOSED'
    STATE_OPEN = 'OPEN'
    STATE_HALF_OPEN = 'HALF_OPEN'

    def __init__(
        self,
        failure_threshold: int = 5,
        timeout: float = 60.0,
        expected_exception: Type[Exception] = Exception
    ):
        """
        Initialize circuit breaker

        Args:
            failure_threshold: Number of failures before opening circuit
            timeout: Seconds to wait before attempting recovery
            expected_exception: Exception type to count as failure
        """
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.expected_exception = expected_exception

        self.failure_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.state = self.STATE_CLOSED

    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Call function through circuit breaker

        Args:
            func: Function to call
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Function result

        Raises:
            CircuitBreakerOpen: If circuit is open
        """
        if self.state == self.STATE_OPEN:
            if self._should_attempt_reset():
                self.state = self.STATE_HALF_OPEN
                logger.info("Circuit breaker entering HALF_OPEN state")
            else:
                raise CircuitBreakerOpen(
                    f"Circuit breaker is OPEN (failures: {self.failure_count})"
                )

        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exception as e:
            self._on_failure()
            raise

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset"""
        if self.last_failure_time is None:
            return True

        return datetime.now() - self.last_failure_time > timedelta(seconds=self.timeout)

    def _on_success(self):
        """Handle successful call"""
        if self.state == self.STATE_HALF_OPEN:
            logger.info("Circuit breaker recovered, returning to CLOSED state")
            self.state = self.STATE_CLOSED
            self.failure_count = 0

    def _on_failure(self):
        """Handle failed call"""
        self.failure_count += 1
        self.last_failure_time = datetime.now()

        if self.state == self.STATE_HALF_OPEN:
            logger.warning("Circuit breaker test failed, returning to OPEN state")
            self.state = self.STATE_OPEN
        elif self.failure_count >= self.failure_threshold:
            logger.error(
                f"Circuit breaker threshold exceeded ({self.failure_count} failures), "
                f"opening circuit"
            )
            self.state = self.STATE_OPEN

    def reset(self):
        """Manually reset circuit breaker"""
        self.failure_count = 0
        self.state = self.STATE_CLOSED
        self.last_failure_time = None
        logger.info("Circuit breaker manually reset")

    def get_state(self) -> dict:
        """Get current state information"""
        return {
            'state': self.state,
            'failure_count': self.failure_count,
            'last_failure': self.last_failure_time.isoformat() if self.last_failure_time else None
        }


class FallbackHandler:
    """
    Graceful degradation with fallback mechanisms

    Usage:
        fallback = FallbackHandler()
        fallback.add_handler(primary_ocr)
        fallback.add_handler(secondary_ocr)
        fallback.add_handler(tesseract_fallback)

        result = fallback.execute(image_path)
    """

    def __init__(self, log_failures: bool = True):
        self.handlers: list[Callable] = []
        self.log_failures = log_failures

    def add_handler(self, handler: Callable):
        """Add a fallback handler"""
        self.handlers.append(handler)

    def execute(self, *args, **kwargs) -> Any:
        """
        Execute handlers in order until one succeeds

        Returns:
            Result from first successful handler

        Raises:
            Exception: If all handlers fail
        """
        last_exception = None

        for i, handler in enumerate(self.handlers, 1):
            try:
                result = handler(*args, **kwargs)
                if i > 1:
                    logger.warning(
                        f"Succeeded with fallback handler #{i} ({handler.__name__})"
                    )
                return result
            except Exception as e:
                last_exception = e
                if self.log_failures:
                    logger.warning(
                        f"Handler #{i} ({handler.__name__}) failed: {e}"
                    )

        logger.error("All fallback handlers exhausted")
        raise last_exception if last_exception else Exception("No handlers available")


class RateLimiter:
    """
    Token bucket rate limiter

    Usage:
        limiter = RateLimiter(rate=10, per=1.0)  # 10 requests per second

        @limiter.limit
        def api_call():
            ...
    """

    def __init__(self, rate: int, per: float = 1.0):
        """
        Initialize rate limiter

        Args:
            rate: Number of tokens
            per: Time period in seconds
        """
        self.rate = rate
        self.per = per
        self.tokens = rate
        self.last_update = time.time()

    def limit(self, func: Callable) -> Callable:
        """Decorator to rate limit function"""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            self._refill()

            if self.tokens < 1:
                sleep_time = (1 - self.tokens) * (self.per / self.rate)
                logger.debug(f"Rate limit reached, sleeping {sleep_time:.2f}s")
                time.sleep(sleep_time)
                self._refill()

            self.tokens -= 1
            return func(*args, **kwargs)

        return wrapper

    def _refill(self):
        """Refill token bucket"""
        now = time.time()
        elapsed = now - self.last_update

        # Add tokens based on elapsed time
        self.tokens = min(
            self.rate,
            self.tokens + (elapsed * (self.rate / self.per))
        )
        self.last_update = now


# Example usage and testing
if __name__ == "__main__":
    # Test retry decorator
    @with_retry(max_attempts=3, initial_delay=0.1)
    def flaky_function(success_on_attempt: int = 2):
        """Simulates a flaky function"""
        if not hasattr(flaky_function, 'attempts'):
            flaky_function.attempts = 0
        flaky_function.attempts += 1

        if flaky_function.attempts < success_on_attempt:
            raise ValueError(f"Attempt {flaky_function.attempts} failed")

        return f"Success on attempt {flaky_function.attempts}"

    # Test circuit breaker
    def test_circuit_breaker():
        breaker = CircuitBreaker(failure_threshold=3, timeout=5.0)

        def failing_service():
            raise Exception("Service unavailable")

        # Fail enough times to open circuit
        for i in range(5):
            try:
                breaker.call(failing_service)
            except Exception as e:
                print(f"Attempt {i+1}: {e}")

        print(f"Circuit state: {breaker.get_state()}")

    # Test fallback
    def test_fallback():
        fallback = FallbackHandler()

        def primary():
            raise Exception("Primary failed")

        def secondary():
            raise Exception("Secondary failed")

        def tertiary():
            return "Tertiary succeeded"

        fallback.add_handler(primary)
        fallback.add_handler(secondary)
        fallback.add_handler(tertiary)

        result = fallback.execute()
        print(f"Result: {result}")

    print("Testing retry...")
    try:
        result = flaky_function(success_on_attempt=2)
        print(f"✓ {result}")
    except RetryExhausted as e:
        print(f"✗ {e}")

    print("\nTesting circuit breaker...")
    test_circuit_breaker()

    print("\nTesting fallback...")
    test_fallback()
