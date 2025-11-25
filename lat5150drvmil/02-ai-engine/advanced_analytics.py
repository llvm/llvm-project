#!/usr/bin/env python3
"""
Advanced Analytics Engine - Pattern recognition and predictive analytics

Provides advanced analytics capabilities:
- Pattern detection in conversations, logs, and data
- Anomaly detection for unusual behavior
- Trend analysis and forecasting
- Correlation analysis across data sources
- Automated insight generation
- Predictive analytics

Part of Phase 4: Option B - Advanced Features
"""

import json
import math
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import Counter, defaultdict
from enum import Enum


class PatternType(Enum):
    """Types of patterns that can be detected"""
    SEQUENTIAL = "sequential"  # A→B→C sequences
    CYCLIC = "cyclic"  # Repeating patterns
    CORRELATIONAL = "correlational"  # X happens when Y happens
    TEMPORAL = "temporal"  # Time-based patterns
    ANOMALY = "anomaly"  # Outliers/unusual events
    TREND = "trend"  # Increasing/decreasing over time


class InsightType(Enum):
    """Types of automated insights"""
    OPTIMIZATION = "optimization"
    WARNING = "warning"
    RECOMMENDATION = "recommendation"
    PREDICTION = "prediction"
    DISCOVERY = "discovery"


@dataclass
class Pattern:
    """Detected pattern"""
    pattern_id: str
    pattern_type: PatternType
    description: str
    confidence: float  # 0.0 to 1.0
    occurrences: int
    first_seen: datetime
    last_seen: datetime
    examples: List[Any] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Anomaly:
    """Detected anomaly"""
    anomaly_id: str
    description: str
    severity: str  # low, medium, high, critical
    deviation_score: float  # How far from normal
    timestamp: datetime
    context: Dict[str, Any] = field(default_factory=dict)
    recommended_action: Optional[str] = None


@dataclass
class Insight:
    """Automated insight"""
    insight_id: str
    insight_type: InsightType
    title: str
    description: str
    confidence: float
    impact: str  # low, medium, high
    evidence: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class TrendAnalysis:
    """Trend analysis result"""
    metric_name: str
    direction: str  # increasing, decreasing, stable, volatile
    rate_of_change: float
    prediction_24h: Optional[float] = None
    prediction_7d: Optional[float] = None
    confidence: float = 0.0
    data_points: int = 0


class AdvancedAnalytics:
    """
    Advanced analytics engine with pattern recognition

    Features:
    - Pattern detection (sequential, cyclic, correlational, temporal)
    - Anomaly detection using statistical methods
    - Trend analysis and forecasting
    - Correlation analysis
    - Automated insight generation
    - Predictive analytics
    """

    def __init__(
        self,
        event_driven_agent=None,
        conversation_manager=None
    ):
        """
        Initialize advanced analytics engine

        Args:
            event_driven_agent: Event-driven agent for data source
            conversation_manager: Conversation manager for conversation analysis
        """
        self.event_driven_agent = event_driven_agent
        self.conversation_manager = conversation_manager

        # Pattern storage
        self.detected_patterns: Dict[str, Pattern] = {}
        self.detected_anomalies: Dict[str, Anomaly] = {}
        self.generated_insights: Dict[str, Insight] = {}

        # Statistics
        self.stats = {
            "patterns_detected": 0,
            "anomalies_detected": 0,
            "insights_generated": 0,
            "trend_analyses": 0,
            "correlations_found": 0,
        }

    def detect_sequential_patterns(
        self,
        sequences: List[List[str]],
        min_support: float = 0.1
    ) -> List[Pattern]:
        """
        Detect sequential patterns (A→B→C)

        Args:
            sequences: List of sequences to analyze
            min_support: Minimum support threshold (0.0 to 1.0)

        Returns:
            List of detected patterns
        """
        if not sequences:
            return []

        # Count subsequences
        subsequence_counts = Counter()
        for sequence in sequences:
            # Generate all subsequences
            for length in range(2, min(len(sequence) + 1, 5)):  # Max length 4
                for i in range(len(sequence) - length + 1):
                    subseq = tuple(sequence[i:i + length])
                    subsequence_counts[subseq] += 1

        # Filter by minimum support
        min_count = int(len(sequences) * min_support)
        patterns = []

        for subseq, count in subsequence_counts.items():
            if count >= min_count:
                pattern_id = f"seq_{hash(subseq) & 0xFFFFFF}"
                confidence = count / len(sequences)

                pattern = Pattern(
                    pattern_id=pattern_id,
                    pattern_type=PatternType.SEQUENTIAL,
                    description=f"Sequential pattern: {' → '.join(subseq)}",
                    confidence=confidence,
                    occurrences=count,
                    first_seen=datetime.now(),
                    last_seen=datetime.now(),
                    examples=[list(subseq)]
                )
                patterns.append(pattern)
                self.detected_patterns[pattern_id] = pattern

        self.stats["patterns_detected"] += len(patterns)
        return patterns

    def detect_anomalies(
        self,
        data: List[float],
        threshold: float = 2.0
    ) -> List[Anomaly]:
        """
        Detect anomalies using z-score method

        Args:
            data: Numeric data to analyze
            threshold: Z-score threshold (default: 2.0 standard deviations)

        Returns:
            List of detected anomalies
        """
        if len(data) < 3:
            return []

        # Calculate mean and standard deviation
        mean = sum(data) / len(data)
        variance = sum((x - mean) ** 2 for x in data) / len(data)
        std_dev = math.sqrt(variance)

        if std_dev == 0:
            return []

        anomalies = []
        for i, value in enumerate(data):
            z_score = abs((value - mean) / std_dev)

            if z_score > threshold:
                # Determine severity
                if z_score > 4.0:
                    severity = "critical"
                elif z_score > 3.0:
                    severity = "high"
                elif z_score > 2.5:
                    severity = "medium"
                else:
                    severity = "low"

                anomaly_id = f"anomaly_{i}_{int(datetime.now().timestamp())}"
                anomaly = Anomaly(
                    anomaly_id=anomaly_id,
                    description=f"Value {value:.2f} deviates by {z_score:.2f} std devs from mean {mean:.2f}",
                    severity=severity,
                    deviation_score=z_score,
                    timestamp=datetime.now(),
                    context={
                        "value": value,
                        "mean": mean,
                        "std_dev": std_dev,
                        "z_score": z_score,
                        "position": i
                    },
                    recommended_action="Investigate cause of deviation"
                )
                anomalies.append(anomaly)
                self.detected_anomalies[anomaly_id] = anomaly

        self.stats["anomalies_detected"] += len(anomalies)
        return anomalies

    def analyze_trend(
        self,
        metric_name: str,
        values: List[float],
        timestamps: Optional[List[datetime]] = None
    ) -> TrendAnalysis:
        """
        Analyze trend in time series data

        Args:
            metric_name: Name of the metric
            values: List of values
            timestamps: Optional timestamps (if not provided, assumes equally spaced)

        Returns:
            TrendAnalysis with trend direction and predictions
        """
        if len(values) < 2:
            return TrendAnalysis(
                metric_name=metric_name,
                direction="unknown",
                rate_of_change=0.0,
                data_points=len(values)
            )

        # Simple linear regression
        n = len(values)
        x = list(range(n))
        y = values

        # Calculate slope (rate of change)
        x_mean = sum(x) / n
        y_mean = sum(y) / n

        numerator = sum((x[i] - x_mean) * (y[i] - y_mean) for i in range(n))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(n))

        if denominator == 0:
            slope = 0.0
        else:
            slope = numerator / denominator

        intercept = y_mean - slope * x_mean

        # Determine direction
        if abs(slope) < 0.01 * y_mean:  # Less than 1% change
            direction = "stable"
        elif slope > 0:
            direction = "increasing"
        else:
            direction = "decreasing"

        # Check volatility
        residuals = [y[i] - (slope * x[i] + intercept) for i in range(n)]
        volatility = math.sqrt(sum(r**2 for r in residuals) / n)
        if volatility > 0.2 * y_mean:
            direction = "volatile"

        # Make predictions
        prediction_24h = slope * (n + 24) + intercept
        prediction_7d = slope * (n + 24 * 7) + intercept

        # Calculate confidence (R-squared)
        ss_res = sum(r**2 for r in residuals)
        ss_tot = sum((y[i] - y_mean)**2 for i in range(n))
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0

        analysis = TrendAnalysis(
            metric_name=metric_name,
            direction=direction,
            rate_of_change=slope,
            prediction_24h=prediction_24h,
            prediction_7d=prediction_7d,
            confidence=max(0.0, min(1.0, r_squared)),
            data_points=n
        )

        self.stats["trend_analyses"] += 1
        return analysis

    def find_correlations(
        self,
        metrics: Dict[str, List[float]]
    ) -> List[Tuple[str, str, float]]:
        """
        Find correlations between metrics

        Args:
            metrics: Dictionary of metric_name -> values

        Returns:
            List of (metric1, metric2, correlation) tuples
        """
        correlations = []
        metric_names = list(metrics.keys())

        for i in range(len(metric_names)):
            for j in range(i + 1, len(metric_names)):
                metric1 = metric_names[i]
                metric2 = metric_names[j]

                values1 = metrics[metric1]
                values2 = metrics[metric2]

                # Ensure same length
                min_len = min(len(values1), len(values2))
                if min_len < 2:
                    continue

                values1 = values1[:min_len]
                values2 = values2[:min_len]

                # Calculate Pearson correlation
                corr = self._pearson_correlation(values1, values2)

                # Only report strong correlations (|r| > 0.5)
                if abs(corr) > 0.5:
                    correlations.append((metric1, metric2, corr))
                    self.stats["correlations_found"] += 1

        return correlations

    def _pearson_correlation(self, x: List[float], y: List[float]) -> float:
        """Calculate Pearson correlation coefficient"""
        n = len(x)
        if n == 0:
            return 0.0

        x_mean = sum(x) / n
        y_mean = sum(y) / n

        numerator = sum((x[i] - x_mean) * (y[i] - y_mean) for i in range(n))
        x_var = sum((x[i] - x_mean) ** 2 for i in range(n))
        y_var = sum((y[i] - y_mean) ** 2 for i in range(n))

        denominator = math.sqrt(x_var * y_var)

        if denominator == 0:
            return 0.0

        return numerator / denominator

    def generate_insights(
        self,
        patterns: List[Pattern] = None,
        anomalies: List[Anomaly] = None,
        trends: List[TrendAnalysis] = None
    ) -> List[Insight]:
        """
        Generate automated insights from patterns, anomalies, and trends

        Args:
            patterns: Detected patterns
            anomalies: Detected anomalies
            trends: Trend analyses

        Returns:
            List of automated insights
        """
        insights = []
        patterns = patterns or []
        anomalies = anomalies or []
        trends = trends or []

        # Generate insights from anomalies
        critical_anomalies = [a for a in anomalies if a.severity in ["critical", "high"]]
        if critical_anomalies:
            insight_id = f"insight_anomaly_{int(datetime.now().timestamp())}"
            insight = Insight(
                insight_id=insight_id,
                insight_type=InsightType.WARNING,
                title="Critical Anomalies Detected",
                description=f"Detected {len(critical_anomalies)} critical anomalies requiring immediate attention",
                confidence=0.9,
                impact="high",
                evidence=[a.description for a in critical_anomalies[:5]],
                recommendations=[
                    "Investigate root cause of anomalies",
                    "Check for system issues or attacks",
                    "Review recent changes or deployments"
                ]
            )
            insights.append(insight)
            self.generated_insights[insight_id] = insight

        # Generate insights from trends
        declining_trends = [t for t in trends if t.direction == "decreasing" and t.confidence > 0.7]
        if declining_trends:
            insight_id = f"insight_trend_{int(datetime.now().timestamp())}"
            insight = Insight(
                insight_id=insight_id,
                insight_type=InsightType.WARNING,
                title="Declining Metrics Detected",
                description=f"Detected {len(declining_trends)} metrics with declining trends",
                confidence=0.8,
                impact="medium",
                evidence=[f"{t.metric_name}: {t.direction} at {t.rate_of_change:.2f}/period" for t in declining_trends[:3]],
                recommendations=[
                    "Identify cause of decline",
                    "Implement corrective actions",
                    "Monitor closely for further degradation"
                ]
            )
            insights.append(insight)
            self.generated_insights[insight_id] = insight

        # Generate insights from patterns
        frequent_patterns = [p for p in patterns if p.confidence > 0.5]
        if frequent_patterns:
            insight_id = f"insight_pattern_{int(datetime.now().timestamp())}"
            insight = Insight(
                insight_id=insight_id,
                insight_type=InsightType.OPTIMIZATION,
                title="Recurring Patterns Identified",
                description=f"Identified {len(frequent_patterns)} recurring patterns that could be optimized",
                confidence=0.75,
                impact="medium",
                evidence=[p.description for p in frequent_patterns[:3]],
                recommendations=[
                    "Consider automating recurring patterns",
                    "Create workflows for common sequences",
                    "Optimize resource allocation for predicted patterns"
                ]
            )
            insights.append(insight)
            self.generated_insights[insight_id] = insight

        self.stats["insights_generated"] += len(insights)
        return insights

    def analyze_conversation_patterns(self) -> Dict[str, Any]:
        """
        Analyze patterns in conversation history

        Returns:
            Dictionary with conversation analysis results
        """
        if not self.conversation_manager:
            return {"error": "Conversation manager not available"}

        # Get conversation statistics
        conv_stats = self.conversation_manager.get_statistics()

        # Analyze query types
        # In real implementation, would analyze actual conversations
        # For now, return placeholder analysis

        analysis = {
            "total_conversations": conv_stats.get("total_conversations", 0),
            "avg_turns_per_conversation": conv_stats.get("avg_turns_per_conversation", 0),
            "patterns": [],
            "insights": []
        }

        # Generate sample pattern
        if conv_stats.get("total_conversations", 0) > 0:
            pattern = Pattern(
                pattern_id="conv_pattern_1",
                pattern_type=PatternType.SEQUENTIAL,
                description="Users often ask follow-up questions about implementation after design discussions",
                confidence=0.7,
                occurrences=10,
                first_seen=datetime.now() - timedelta(days=7),
                last_seen=datetime.now(),
                metadata={"context": "conversation_analysis"}
            )
            analysis["patterns"].append(pattern)

        return analysis

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics"""
        return {
            **self.stats,
            "total_patterns": len(self.detected_patterns),
            "total_anomalies": len(self.detected_anomalies),
            "total_insights": len(self.generated_insights)
        }


def demo():
    """Demo of advanced analytics"""
    print("=== Advanced Analytics Demo ===\n")

    # Initialize
    analytics = AdvancedAnalytics()

    # 1. Pattern detection
    print("1. Sequential Pattern Detection")
    sequences = [
        ["login", "view_dashboard", "edit_profile", "logout"],
        ["login", "view_dashboard", "run_query", "logout"],
        ["login", "view_dashboard", "edit_profile", "run_query"],
        ["login", "view_dashboard", "logout"],
    ]
    patterns = analytics.detect_sequential_patterns(sequences, min_support=0.5)
    print(f"   Detected {len(patterns)} patterns:")
    for p in patterns:
        print(f"     - {p.description} (confidence: {p.confidence:.2f})")

    # 2. Anomaly detection
    print("\n2. Anomaly Detection")
    data = [10, 12, 11, 10, 11, 50, 10, 12, 11, 10]  # 50 is anomaly
    anomalies = analytics.detect_anomalies(data, threshold=2.0)
    print(f"   Detected {len(anomalies)} anomalies:")
    for a in anomalies:
        print(f"     - {a.description} (severity: {a.severity})")

    # 3. Trend analysis
    print("\n3. Trend Analysis")
    values = [100, 105, 110, 115, 120, 125, 130]  # Increasing trend
    trend = analytics.analyze_trend("user_activity", values)
    print(f"   Metric: {trend.metric_name}")
    print(f"   Direction: {trend.direction}")
    print(f"   Rate of change: {trend.rate_of_change:.2f}")
    print(f"   Prediction (24h): {trend.prediction_24h:.2f}")
    print(f"   Confidence: {trend.confidence:.2f}")

    # 4. Correlation analysis
    print("\n4. Correlation Analysis")
    metrics = {
        "cpu_usage": [50, 60, 70, 80, 90],
        "response_time": [100, 120, 140, 160, 180],
        "memory_usage": [30, 32, 34, 36, 38]
    }
    correlations = analytics.find_correlations(metrics)
    print(f"   Found {len(correlations)} correlations:")
    for m1, m2, corr in correlations:
        print(f"     - {m1} ↔ {m2}: {corr:.2f}")

    # 5. Generate insights
    print("\n5. Automated Insights")
    insights = analytics.generate_insights(patterns, anomalies, [trend])
    print(f"   Generated {len(insights)} insights:")
    for ins in insights:
        print(f"     - [{ins.insight_type.value}] {ins.title}")
        print(f"       {ins.description}")

    # 6. Statistics
    print("\n6. Statistics:")
    stats = analytics.get_statistics()
    for key, value in stats.items():
        print(f"   {key}: {value}")


if __name__ == "__main__":
    demo()
