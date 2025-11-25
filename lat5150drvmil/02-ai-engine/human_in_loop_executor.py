#!/usr/bin/env python3
"""
Human-in-Loop Executor
Production safety feature for AI-driven operations

Key Capabilities:
- Approval workflow for sensitive operations
- Risk assessment and classification
- Audit logging of human decisions
- Configurable approval thresholds
- Async approval with timeout

Use Cases:
- Financial transactions requiring approval
- Data deletion or modification
- External API calls with side effects
- Security-sensitive operations
- Compliance-required approvals
"""

import asyncio
import hashlib
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
import json


class RiskLevel(Enum):
    """Risk level classification"""
    LOW = "low"  # Auto-approve
    MEDIUM = "medium"  # Request approval
    HIGH = "high"  # Request approval + additional context
    CRITICAL = "critical"  # Require explicit justification


class ApprovalStatus(Enum):
    """Approval status"""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    TIMEOUT = "timeout"
    AUTO_APPROVED = "auto_approved"


@dataclass
class ApprovalRequest:
    """
    Request for human approval

    Attributes:
        request_id: Unique request identifier
        operation: Operation name/description
        parameters: Operation parameters
        risk_level: Assessed risk level
        risk_reasoning: Why this risk level
        requested_at: Timestamp of request
        status: Current approval status
        approved_by: Who approved (if approved)
        approved_at: When approved
        rejection_reason: Reason for rejection
        timeout_seconds: Timeout for approval
    """
    request_id: str
    operation: str
    parameters: Dict[str, Any]
    risk_level: RiskLevel
    risk_reasoning: str
    requested_at: datetime = field(default_factory=datetime.now)
    status: ApprovalStatus = ApprovalStatus.PENDING
    approved_by: Optional[str] = None
    approved_at: Optional[datetime] = None
    rejection_reason: Optional[str] = None
    timeout_seconds: int = 300  # 5 minutes default

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "request_id": self.request_id,
            "operation": self.operation,
            "parameters": self.parameters,
            "risk_level": self.risk_level.value,
            "risk_reasoning": self.risk_reasoning,
            "requested_at": self.requested_at.isoformat(),
            "status": self.status.value,
            "approved_by": self.approved_by,
            "approved_at": self.approved_at.isoformat() if self.approved_at else None,
            "rejection_reason": self.rejection_reason,
            "timeout_seconds": self.timeout_seconds
        }


@dataclass
class ExecutionResult:
    """
    Result of human-in-loop execution

    Attributes:
        success: Whether execution succeeded
        result: Execution result (if successful)
        error: Error message (if failed)
        approval_request: Associated approval request
        execution_time_ms: Execution time in milliseconds
    """
    success: bool
    result: Any = None
    error: Optional[str] = None
    approval_request: Optional[ApprovalRequest] = None
    execution_time_ms: int = 0


class HumanInLoopExecutor:
    """
    Execute operations with human approval for sensitive actions

    Features:
    - Automatic risk assessment
    - Configurable approval thresholds
    - Timeout handling
    - Audit logging
    - Approval history
    """

    def __init__(
        self,
        auto_approve_low_risk: bool = True,
        default_timeout_seconds: int = 300,
        audit_log_path: Optional[str] = None
    ):
        """
        Initialize human-in-loop executor

        Args:
            auto_approve_low_risk: Auto-approve low-risk operations
            default_timeout_seconds: Default timeout for approvals
            audit_log_path: Path to audit log file
        """
        self.auto_approve_low_risk = auto_approve_low_risk
        self.default_timeout_seconds = default_timeout_seconds
        self.audit_log_path = Path(audit_log_path) if audit_log_path else None

        # Approval tracking
        self.pending_approvals: Dict[str, ApprovalRequest] = {}
        self.approval_history: List[ApprovalRequest] = []

        # Risk patterns
        self.risk_patterns = self._initialize_risk_patterns()

        # Approval callbacks
        self.approval_callbacks: Dict[str, asyncio.Future] = {}

    def _initialize_risk_patterns(self) -> Dict[str, RiskLevel]:
        """
        Initialize risk patterns for operations

        Returns:
            Dict mapping operation patterns to risk levels
        """
        return {
            # Low risk - read-only operations
            "query": RiskLevel.LOW,
            "search": RiskLevel.LOW,
            "get": RiskLevel.LOW,
            "list": RiskLevel.LOW,
            "read": RiskLevel.LOW,

            # Medium risk - writes without deletion
            "create": RiskLevel.MEDIUM,
            "update": RiskLevel.MEDIUM,
            "modify": RiskLevel.MEDIUM,
            "write": RiskLevel.MEDIUM,

            # High risk - deletions and external calls
            "delete": RiskLevel.HIGH,
            "remove": RiskLevel.HIGH,
            "drop": RiskLevel.HIGH,
            "api_call": RiskLevel.HIGH,
            "external": RiskLevel.HIGH,

            # Critical - financial and security
            "transfer": RiskLevel.CRITICAL,
            "payment": RiskLevel.CRITICAL,
            "financial": RiskLevel.CRITICAL,
            "security": RiskLevel.CRITICAL,
            "admin": RiskLevel.CRITICAL
        }

    async def execute(
        self,
        operation: str,
        operation_func: Callable,
        parameters: Dict[str, Any],
        risk_override: Optional[RiskLevel] = None,
        timeout_seconds: Optional[int] = None
    ) -> ExecutionResult:
        """
        Execute operation with human-in-loop approval if needed

        Args:
            operation: Operation name/description
            operation_func: Callable to execute
            parameters: Parameters for the operation
            risk_override: Optional risk level override
            timeout_seconds: Optional timeout override

        Returns:
            ExecutionResult with approval and execution info
        """
        start_time = datetime.now()

        # Step 1: Assess risk
        risk_level, risk_reasoning = self._assess_risk(operation, parameters, risk_override)

        # Step 2: Check if approval needed
        if self._needs_approval(risk_level):
            # Create approval request
            request = ApprovalRequest(
                request_id=self._generate_request_id(operation, parameters),
                operation=operation,
                parameters=parameters,
                risk_level=risk_level,
                risk_reasoning=risk_reasoning,
                timeout_seconds=timeout_seconds or self.default_timeout_seconds
            )

            # Request approval
            approval_granted = await self._request_approval(request)

            if not approval_granted:
                # Approval denied or timeout
                self._log_audit(request)
                execution_time = int((datetime.now() - start_time).total_seconds() * 1000)

                return ExecutionResult(
                    success=False,
                    error=f"Approval {request.status.value}: {request.rejection_reason or 'Not approved'}",
                    approval_request=request,
                    execution_time_ms=execution_time
                )

        else:
            # Auto-approve low risk
            request = ApprovalRequest(
                request_id=self._generate_request_id(operation, parameters),
                operation=operation,
                parameters=parameters,
                risk_level=risk_level,
                risk_reasoning=risk_reasoning,
                status=ApprovalStatus.AUTO_APPROVED
            )

        # Step 3: Execute operation
        try:
            if asyncio.iscoroutinefunction(operation_func):
                result = await operation_func(**parameters)
            else:
                result = operation_func(**parameters)

            execution_time = int((datetime.now() - start_time).total_seconds() * 1000)

            # Log successful execution
            self._log_audit(request)

            return ExecutionResult(
                success=True,
                result=result,
                approval_request=request,
                execution_time_ms=execution_time
            )

        except Exception as e:
            execution_time = int((datetime.now() - start_time).total_seconds() * 1000)

            return ExecutionResult(
                success=False,
                error=str(e),
                approval_request=request,
                execution_time_ms=execution_time
            )

    def _assess_risk(
        self,
        operation: str,
        parameters: Dict[str, Any],
        risk_override: Optional[RiskLevel]
    ) -> tuple[RiskLevel, str]:
        """
        Assess risk level of operation

        Args:
            operation: Operation name
            parameters: Operation parameters
            risk_override: Optional override

        Returns:
            Tuple of (risk_level, reasoning)
        """
        if risk_override:
            return risk_override, "Risk level manually overridden"

        operation_lower = operation.lower()
        reasoning = []

        # Check operation patterns
        matched_risk = RiskLevel.MEDIUM  # Default
        for pattern, risk_level in self.risk_patterns.items():
            if pattern in operation_lower:
                matched_risk = risk_level
                reasoning.append(f"Operation contains '{pattern}' keyword")
                break

        # Check parameter-based risk escalation
        if "amount" in parameters or "value" in parameters:
            # Financial parameters increase risk
            if matched_risk == RiskLevel.MEDIUM:
                matched_risk = RiskLevel.HIGH
            reasoning.append("Contains financial parameters")

        if "all" in str(parameters).lower() or "wildcard" in str(parameters).lower():
            # Bulk operations increase risk
            if matched_risk.value in ["low", "medium"]:
                matched_risk = RiskLevel.HIGH
            reasoning.append("Bulk operation detected")

        reasoning_text = "; ".join(reasoning) if reasoning else "Default risk assessment"

        return matched_risk, reasoning_text

    def _needs_approval(self, risk_level: RiskLevel) -> bool:
        """Check if risk level requires approval"""
        if risk_level == RiskLevel.LOW and self.auto_approve_low_risk:
            return False
        return True

    async def _request_approval(self, request: ApprovalRequest) -> bool:
        """
        Request human approval

        Args:
            request: Approval request

        Returns:
            True if approved, False otherwise
        """
        # Store pending request
        self.pending_approvals[request.request_id] = request

        # Create future for approval callback
        approval_future = asyncio.Future()
        self.approval_callbacks[request.request_id] = approval_future

        # Print approval request (in production, this would go to UI/notification)
        self._print_approval_request(request)

        # Wait for approval or timeout
        try:
            approved = await asyncio.wait_for(
                approval_future,
                timeout=request.timeout_seconds
            )
            return approved

        except asyncio.TimeoutError:
            request.status = ApprovalStatus.TIMEOUT
            request.rejection_reason = f"Approval timeout after {request.timeout_seconds}s"
            return False

        finally:
            # Cleanup
            self.pending_approvals.pop(request.request_id, None)
            self.approval_callbacks.pop(request.request_id, None)
            self.approval_history.append(request)

    def approve(
        self,
        request_id: str,
        approved_by: str = "human",
        justification: Optional[str] = None
    ) -> bool:
        """
        Approve a pending request

        Args:
            request_id: Request ID to approve
            approved_by: Who approved
            justification: Optional approval justification

        Returns:
            True if approval recorded, False if request not found
        """
        if request_id not in self.pending_approvals:
            return False

        request = self.pending_approvals[request_id]
        request.status = ApprovalStatus.APPROVED
        request.approved_by = approved_by
        request.approved_at = datetime.now()

        # Notify waiting coroutine
        if request_id in self.approval_callbacks:
            self.approval_callbacks[request_id].set_result(True)

        return True

    def reject(
        self,
        request_id: str,
        reason: str = "Rejected by human reviewer"
    ) -> bool:
        """
        Reject a pending request

        Args:
            request_id: Request ID to reject
            reason: Rejection reason

        Returns:
            True if rejection recorded, False if request not found
        """
        if request_id not in self.pending_approvals:
            return False

        request = self.pending_approvals[request_id]
        request.status = ApprovalStatus.REJECTED
        request.rejection_reason = reason

        # Notify waiting coroutine
        if request_id in self.approval_callbacks:
            self.approval_callbacks[request_id].set_result(False)

        return True

    def _generate_request_id(self, operation: str, parameters: Dict) -> str:
        """Generate unique request ID"""
        content = f"{operation}:{json.dumps(parameters, sort_keys=True)}:{datetime.now().isoformat()}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def _print_approval_request(self, request: ApprovalRequest):
        """Print approval request (in production, send to UI)"""
        print("\n" + "="*70)
        print("🚨 APPROVAL REQUIRED")
        print("="*70)
        print(f"Request ID: {request.request_id}")
        print(f"Operation: {request.operation}")
        print(f"Risk Level: {request.risk_level.value.upper()}")
        print(f"Risk Reasoning: {request.risk_reasoning}")
        print(f"\nParameters:")
        for key, value in request.parameters.items():
            print(f"  {key}: {value}")
        print(f"\nTimeout: {request.timeout_seconds}s")
        print("\nTo approve: executor.approve('{0}')".format(request.request_id))
        print("To reject:  executor.reject('{0}')".format(request.request_id))
        print("="*70 + "\n")

    def _log_audit(self, request: ApprovalRequest):
        """Log approval to audit trail"""
        if not self.audit_log_path:
            return

        try:
            # Ensure directory exists
            self.audit_log_path.parent.mkdir(parents=True, exist_ok=True)

            # Append to audit log
            with open(self.audit_log_path, 'a') as f:
                f.write(json.dumps(request.to_dict()) + "\n")

        except Exception as e:
            print(f"⚠️  Failed to write audit log: {e}")

    def get_pending_requests(self) -> List[ApprovalRequest]:
        """Get all pending approval requests"""
        return list(self.pending_approvals.values())

    def get_approval_history(
        self,
        limit: int = 100,
        status: Optional[ApprovalStatus] = None
    ) -> List[ApprovalRequest]:
        """
        Get approval history

        Args:
            limit: Maximum number of entries
            status: Optional filter by status

        Returns:
            List of approval requests
        """
        history = self.approval_history

        if status:
            history = [r for r in history if r.status == status]

        return history[-limit:]

    def get_statistics(self) -> Dict[str, Any]:
        """Get executor statistics"""
        status_counts = {}
        risk_counts = {}

        for request in self.approval_history:
            status = request.status.value
            risk = request.risk_level.value

            status_counts[status] = status_counts.get(status, 0) + 1
            risk_counts[risk] = risk_counts.get(risk, 0) + 1

        return {
            "total_requests": len(self.approval_history),
            "pending_requests": len(self.pending_approvals),
            "status_breakdown": status_counts,
            "risk_breakdown": risk_counts,
            "auto_approve_enabled": self.auto_approve_low_risk,
            "default_timeout_seconds": self.default_timeout_seconds
        }


async def main():
    """Demo usage"""
    print("=== Human-in-Loop Executor Demo ===\n")

    executor = HumanInLoopExecutor(
        auto_approve_low_risk=True,
        default_timeout_seconds=10,
        audit_log_path="/tmp/hilp_audit.log"
    )

    # Demo operations
    async def safe_query(query: str):
        """Safe read operation"""
        return f"Query result for: {query}"

    async def risky_delete(resource_id: str):
        """Risky delete operation"""
        return f"Deleted resource: {resource_id}"

    # Test 1: Low-risk operation (auto-approved)
    print("1. Low-risk operation (auto-approved):")
    result = await executor.execute(
        operation="query_database",
        operation_func=safe_query,
        parameters={"query": "SELECT * FROM users LIMIT 10"}
    )
    print(f"   Result: {result.success}")
    print(f"   Status: {result.approval_request.status.value}")
    print(f"   Time: {result.execution_time_ms}ms\n")

    # Test 2: High-risk operation (requires approval)
    print("2. High-risk operation (requires approval):")

    # Start execution in background
    exec_task = asyncio.create_task(
        executor.execute(
            operation="delete_resource",
            operation_func=risky_delete,
            parameters={"resource_id": "user_12345"},
            timeout_seconds=5
        )
    )

    # Simulate approval after 1 second
    await asyncio.sleep(1)

    pending = executor.get_pending_requests()
    if pending:
        print(f"   Approving request: {pending[0].request_id}")
        executor.approve(pending[0].request_id, approved_by="admin")

    result = await exec_task
    print(f"   Result: {result.success}")
    print(f"   Status: {result.approval_request.status.value}")
    print(f"   Approved by: {result.approval_request.approved_by}\n")

    # Statistics
    print("3. Statistics:")
    stats = executor.get_statistics()
    print(f"   Total requests: {stats['total_requests']}")
    print(f"   Status breakdown: {stats['status_breakdown']}")
    print(f"   Risk breakdown: {stats['risk_breakdown']}")


if __name__ == "__main__":
    asyncio.run(main())
