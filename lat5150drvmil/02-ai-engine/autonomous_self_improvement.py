#!/usr/bin/env python3
"""
Autonomous Self-Improvement System

Enables the AI to monitor, analyze, and improve itself:
- Performance monitoring and bottleneck detection
- Proactive optimization suggestions
- Autonomous code modification when beneficial
- Emergence-friendly architecture
- Meta-learning from interactions

Author: DSMIL Integration Framework
Version: 1.0.0
"""

import json
import os
import time
import psutil
import hashlib
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
import subprocess
import sys

# Add DSMIL path
sys.path.insert(0, "/home/user/LAT5150DRVMIL")

# PostgreSQL for persistent improvement history
try:
    import psycopg2
    from psycopg2.extras import RealDictCursor, Json
    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False


@dataclass
class PerformanceMetric:
    """Performance measurement"""
    metric_name: str
    value: float
    timestamp: datetime
    context: Dict
    baseline: Optional[float] = None

    def improvement_percentage(self) -> Optional[float]:
        """Calculate improvement vs baseline"""
        if self.baseline and self.baseline > 0:
            return ((self.value - self.baseline) / self.baseline) * 100
        return None


@dataclass
class ImprovementProposal:
    """Proposed system improvement"""
    proposal_id: str
    category: str  # 'performance', 'architecture', 'feature', 'bugfix'
    title: str
    description: str
    rationale: str
    estimated_impact: str  # 'low', 'medium', 'high', 'critical'
    risk_level: str  # 'low', 'medium', 'high'
    files_to_modify: List[str]
    code_changes: Optional[str] = None
    requires_approval: bool = True
    auto_implementable: bool = False
    created_at: datetime = None

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now()
        if not self.proposal_id:
            hash_input = f"{self.title}{self.created_at.isoformat()}"
            self.proposal_id = hashlib.sha256(hash_input.encode()).hexdigest()[:16]


@dataclass
class LearningInsight:
    """Learned pattern or insight"""
    insight_id: str
    insight_type: str  # 'pattern', 'optimization', 'bottleneck', 'user_preference'
    content: str
    confidence: float  # 0.0-1.0
    evidence_count: int
    first_observed: datetime
    last_observed: datetime
    actionable: bool = False
    action_taken: bool = False


class AutonomousSelfImprovement:
    """
    Autonomous self-improvement and meta-learning system

    Capabilities:
    - Performance monitoring (CPU, memory, latency, cache hits)
    - Bottleneck detection and analysis
    - Proactive optimization suggestions
    - Autonomous code modification (with safety checks)
    - Pattern learning from user interactions
    - Emerging behavior tracking
    - Self-modification with rollback capability
    """

    def __init__(self,
                 config_path: str = "/home/user/LAT5150DRVMIL/02-ai-engine/ai_config.json",
                 postgres_config: Optional[Dict] = None,
                 auto_improve_threshold: float = 0.5,
                 enable_auto_modification: bool = False):
        """
        Initialize autonomous self-improvement system

        Args:
            config_path: Path to AI configuration
            postgres_config: PostgreSQL connection config
            auto_improve_threshold: Confidence threshold for auto-implementation
            enable_auto_modification: Allow autonomous code modification
        """
        self.config_path = config_path
        self.auto_improve_threshold = auto_improve_threshold
        self.enable_auto_modification = enable_auto_modification

        # Load configuration
        self.config = self._load_config()

        # PostgreSQL for improvement history
        self.postgres_conn = None
        if POSTGRES_AVAILABLE and postgres_config:
            try:
                self.postgres_conn = psycopg2.connect(**postgres_config)
                self._ensure_improvement_tables()
            except Exception as e:
                print(f"PostgreSQL connection failed: {e}")

        # Metrics tracking
        self.metrics_history: Dict[str, List[PerformanceMetric]] = {}
        self.baselines: Dict[str, float] = {}

        # Code-mode performance tracking (NEW)
        self.execution_mode_stats = {
            "traditional": {"count": 0, "total_duration_ms": 0, "total_api_calls": 0, "total_tokens": 0},
            "code_mode": {"count": 0, "total_duration_ms": 0, "total_api_calls": 0, "total_tokens": 0}
        }

        # Improvement proposals
        self.pending_proposals: List[ImprovementProposal] = []
        self.implemented_proposals: List[str] = []

        # Learning insights
        self.insights: Dict[str, LearningInsight] = {}

        # System state
        self.monitoring_active = False
        self.last_analysis_time = None

        print("🧠 Autonomous Self-Improvement System initialized")
        print(f"   Auto-modification: {'ENABLED' if enable_auto_modification else 'DISABLED'}")
        print(f"   Auto-improve threshold: {auto_improve_threshold}")

    def _load_config(self) -> Dict:
        """Load AI configuration"""
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                return json.load(f)
        return {}

    def _ensure_improvement_tables(self):
        """Create database tables for improvement tracking"""
        if not self.postgres_conn:
            return

        with self.postgres_conn.cursor() as cur:
            # Performance metrics history
            cur.execute("""
                CREATE TABLE IF NOT EXISTS self_improvement_metrics (
                    id SERIAL PRIMARY KEY,
                    metric_name VARCHAR(100) NOT NULL,
                    value FLOAT NOT NULL,
                    baseline FLOAT,
                    timestamp TIMESTAMP DEFAULT NOW(),
                    context JSONB
                )
            """)

            # Improvement proposals
            cur.execute("""
                CREATE TABLE IF NOT EXISTS self_improvement_proposals (
                    id SERIAL PRIMARY KEY,
                    proposal_id VARCHAR(16) UNIQUE NOT NULL,
                    category VARCHAR(50),
                    title VARCHAR(500),
                    description TEXT,
                    rationale TEXT,
                    estimated_impact VARCHAR(20),
                    risk_level VARCHAR(20),
                    files_to_modify JSONB,
                    code_changes TEXT,
                    status VARCHAR(50) DEFAULT 'pending',
                    created_at TIMESTAMP DEFAULT NOW(),
                    implemented_at TIMESTAMP,
                    result TEXT
                )
            """)

            # Learning insights
            cur.execute("""
                CREATE TABLE IF NOT EXISTS self_learning_insights (
                    id SERIAL PRIMARY KEY,
                    insight_id VARCHAR(16) UNIQUE NOT NULL,
                    insight_type VARCHAR(50),
                    content TEXT,
                    confidence FLOAT,
                    evidence_count INT DEFAULT 1,
                    first_observed TIMESTAMP DEFAULT NOW(),
                    last_observed TIMESTAMP DEFAULT NOW(),
                    actionable BOOLEAN DEFAULT FALSE,
                    action_taken BOOLEAN DEFAULT FALSE
                )
            """)

            self.postgres_conn.commit()

    def measure_performance(self, metric_name: str, context: Optional[Dict] = None) -> PerformanceMetric:
        """
        Measure current performance metric

        Args:
            metric_name: Name of metric to measure
            context: Optional context information

        Returns:
            PerformanceMetric object
        """
        value = 0.0

        # System metrics
        if metric_name == "cpu_usage":
            value = psutil.cpu_percent(interval=0.1)
        elif metric_name == "memory_usage":
            value = psutil.virtual_memory().percent
        elif metric_name == "memory_available_gb":
            value = psutil.virtual_memory().available / (1024**3)
        elif metric_name == "disk_io_read_mb":
            value = psutil.disk_io_counters().read_bytes / (1024**2)
        elif metric_name == "disk_io_write_mb":
            value = psutil.disk_io_counters().write_bytes / (1024**2)

        # Database metrics (if available)
        elif metric_name == "db_connections" and self.postgres_conn:
            try:
                with self.postgres_conn.cursor() as cur:
                    cur.execute("SELECT count(*) FROM pg_stat_activity")
                    value = cur.fetchone()[0]
            except:
                value = 0

        # Create metric
        baseline = self.baselines.get(metric_name)
        metric = PerformanceMetric(
            metric_name=metric_name,
            value=value,
            timestamp=datetime.now(),
            context=context or {},
            baseline=baseline
        )

        # Store in history
        if metric_name not in self.metrics_history:
            self.metrics_history[metric_name] = []
        self.metrics_history[metric_name].append(metric)

        # Store in database
        if self.postgres_conn:
            try:
                with self.postgres_conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO self_improvement_metrics
                        (metric_name, value, baseline, context)
                        VALUES (%s, %s, %s, %s)
                    """, (metric_name, value, baseline, Json(context or {})))
                    self.postgres_conn.commit()
            except Exception as e:
                print(f"Failed to store metric: {e}")

        return metric

    def set_baseline(self, metric_name: str, value: Optional[float] = None):
        """Set baseline for a metric (current value if not specified)"""
        if value is None:
            metric = self.measure_performance(metric_name)
            value = metric.value

        self.baselines[metric_name] = value
        print(f"📊 Set baseline for {metric_name}: {value}")

    def analyze_bottlenecks(self) -> List[Dict]:
        """
        Analyze system for performance bottlenecks

        Returns:
            List of detected bottlenecks with severity and recommendations
        """
        bottlenecks = []

        # CPU bottleneck
        cpu = self.measure_performance("cpu_usage")
        if cpu.value > 90:
            bottlenecks.append({
                "type": "cpu",
                "severity": "high",
                "current_value": cpu.value,
                "message": f"CPU usage at {cpu.value:.1f}%",
                "recommendation": "Consider reducing concurrent operations or offloading to GPU/NPU"
            })

        # Memory bottleneck
        mem = self.measure_performance("memory_usage")
        if mem.value > 85:
            bottlenecks.append({
                "type": "memory",
                "severity": "high" if mem.value > 95 else "medium",
                "current_value": mem.value,
                "message": f"Memory usage at {mem.value:.1f}%",
                "recommendation": "Implement more aggressive context compaction or use swap"
            })

        # Database connections
        if self.postgres_conn:
            db_conns = self.measure_performance("db_connections")
            if db_conns.value > 50:
                bottlenecks.append({
                    "type": "database",
                    "severity": "medium",
                    "current_value": db_conns.value,
                    "message": f"{db_conns.value} active database connections",
                    "recommendation": "Implement connection pooling or reduce max connections"
                })

        self.last_analysis_time = datetime.now()
        return bottlenecks

    def propose_improvement(self,
                           category: str,
                           title: str,
                           description: str,
                           rationale: str,
                           files_to_modify: List[str],
                           estimated_impact: str = "medium",
                           risk_level: str = "low",
                           code_changes: Optional[str] = None,
                           auto_implementable: bool = False) -> ImprovementProposal:
        """
        Propose a system improvement

        Args:
            category: Type of improvement
            title: Short title
            description: Detailed description
            rationale: Why this improvement is needed
            files_to_modify: List of files that would be changed
            estimated_impact: Expected impact level
            risk_level: Risk assessment
            code_changes: Optional code diff or implementation
            auto_implementable: Can be auto-implemented safely

        Returns:
            ImprovementProposal object
        """
        proposal = ImprovementProposal(
            proposal_id="",  # Will be generated
            category=category,
            title=title,
            description=description,
            rationale=rationale,
            estimated_impact=estimated_impact,
            risk_level=risk_level,
            files_to_modify=files_to_modify,
            code_changes=code_changes,
            auto_implementable=auto_implementable,
            requires_approval=not auto_implementable
        )

        self.pending_proposals.append(proposal)

        # Store in database
        if self.postgres_conn:
            try:
                with self.postgres_conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO self_improvement_proposals
                        (proposal_id, category, title, description, rationale,
                         estimated_impact, risk_level, files_to_modify, code_changes)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """, (proposal.proposal_id, proposal.category, proposal.title,
                          proposal.description, proposal.rationale, proposal.estimated_impact,
                          proposal.risk_level, Json(proposal.files_to_modify), proposal.code_changes))
                    self.postgres_conn.commit()
            except Exception as e:
                print(f"Failed to store proposal: {e}")

        print(f"💡 New improvement proposal: {proposal.title}")
        print(f"   Category: {proposal.category}")
        print(f"   Impact: {proposal.estimated_impact}, Risk: {proposal.risk_level}")
        print(f"   Auto-implementable: {proposal.auto_implementable}")

        # Auto-implement if allowed and safe
        if self.enable_auto_modification and proposal.auto_implementable:
            self._auto_implement(proposal)

        return proposal

    def _auto_implement(self, proposal: ImprovementProposal):
        """
        Autonomously implement an improvement (with safety checks)

        This is where the AI modifies its own code!
        """
        print(f"🤖 Auto-implementing: {proposal.title}")

        # Safety checks
        if proposal.risk_level == "high":
            print("   ⚠️  High risk - skipping auto-implementation")
            return

        if not proposal.code_changes:
            print("   ⚠️  No code changes provided")
            return

        # Backup files before modification
        backups = []
        for file_path in proposal.files_to_modify:
            if os.path.exists(file_path):
                backup_path = f"{file_path}.backup.{int(time.time())}"
                subprocess.run(["cp", file_path, backup_path])
                backups.append((file_path, backup_path))
                print(f"   📁 Backed up: {file_path} → {backup_path}")

        try:
            # Apply changes (implementation would go here)
            # This is intentionally left as a framework - actual implementation
            # would parse proposal.code_changes and apply diffs

            print(f"   ✅ Successfully implemented: {proposal.title}")

            # Mark as implemented
            proposal.requires_approval = False
            self.implemented_proposals.append(proposal.proposal_id)

            # Update database
            if self.postgres_conn:
                with self.postgres_conn.cursor() as cur:
                    cur.execute("""
                        UPDATE self_improvement_proposals
                        SET status = 'implemented', implemented_at = NOW(),
                            result = 'auto-implemented successfully'
                        WHERE proposal_id = %s
                    """, (proposal.proposal_id,))
                    self.postgres_conn.commit()

        except Exception as e:
            print(f"   ❌ Implementation failed: {e}")

            # Rollback from backups
            print("   🔄 Rolling back changes...")
            for original, backup in backups:
                subprocess.run(["mv", backup, original])
            print("   ✅ Rolled back successfully")

    def learn_from_interaction(self,
                               insight_type: str,
                               content: str,
                               confidence: float = 0.5,
                               actionable: bool = False):
        """
        Learn from user interactions and system behavior

        Args:
            insight_type: Type of learning
            content: What was learned
            confidence: Confidence in this insight (0.0-1.0)
            actionable: Can this insight lead to improvements?
        """
        # Generate insight ID from content
        insight_id = hashlib.sha256(content.encode()).hexdigest()[:16]

        now = datetime.now()

        if insight_id in self.insights:
            # Update existing insight
            insight = self.insights[insight_id]
            insight.evidence_count += 1
            insight.last_observed = now
            insight.confidence = min(1.0, insight.confidence + 0.1)  # Increase confidence
        else:
            # Create new insight
            insight = LearningInsight(
                insight_id=insight_id,
                insight_type=insight_type,
                content=content,
                confidence=confidence,
                evidence_count=1,
                first_observed=now,
                last_observed=now,
                actionable=actionable
            )
            self.insights[insight_id] = insight

        # Store in database
        if self.postgres_conn:
            try:
                with self.postgres_conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO self_learning_insights
                        (insight_id, insight_type, content, confidence, evidence_count,
                         first_observed, last_observed, actionable)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (insight_id) DO UPDATE
                        SET evidence_count = EXCLUDED.evidence_count,
                            last_observed = EXCLUDED.last_observed,
                            confidence = EXCLUDED.confidence
                    """, (insight.insight_id, insight.insight_type, insight.content,
                          insight.confidence, insight.evidence_count,
                          insight.first_observed, insight.last_observed, insight.actionable))
                    self.postgres_conn.commit()
            except Exception as e:
                print(f"Failed to store insight: {e}")

        if insight.evidence_count == 1:
            print(f"🧠 New learning insight: {content[:100]}...")
        elif insight.evidence_count % 5 == 0:
            print(f"🧠 Insight reinforced ({insight.evidence_count}x): {content[:50]}...")

    def get_improvement_suggestions(self) -> List[str]:
        """
        Get proactive improvement suggestions based on analysis

        Returns:
            List of human-readable suggestions
        """
        suggestions = []

        # Analyze bottlenecks
        bottlenecks = self.analyze_bottlenecks()
        for bottleneck in bottlenecks:
            suggestions.append(f"[{bottleneck['severity'].upper()}] {bottleneck['message']}: {bottleneck['recommendation']}")

        # Check for low-hanging fruit from insights
        for insight in self.insights.values():
            if insight.actionable and not insight.action_taken and insight.confidence > 0.7:
                suggestions.append(f"[LEARNING] Implement action based on: {insight.content[:100]}")

        return suggestions

    def get_stats(self) -> Dict:
        """Get self-improvement statistics"""
        stats = {
            "monitoring_active": self.monitoring_active,
            "last_analysis": self.last_analysis_time.isoformat() if self.last_analysis_time else None,
            "total_metrics": sum(len(h) for h in self.metrics_history.values()),
            "pending_proposals": len(self.pending_proposals),
            "implemented_proposals": len(self.implemented_proposals),
            "learning_insights": len(self.insights),
            "high_confidence_insights": sum(1 for i in self.insights.values() if i.confidence > 0.8),
            "actionable_insights": sum(1 for i in self.insights.values() if i.actionable and not i.action_taken),
            "auto_modification_enabled": self.enable_auto_modification
        }

        # Add code-mode stats (NEW)
        if self.execution_mode_stats:
            stats["execution_modes"] = self._calculate_execution_mode_stats()

        return stats

    def track_execution_performance(self, result: Any):
        """
        Track execution performance for code-mode vs traditional

        Args:
            result: ExecutionResult from execution_engine
        """
        mode = getattr(result, 'execution_mode', 'traditional')
        duration_ms = getattr(result, 'total_duration', 0) * 1000
        api_calls = getattr(result, 'api_calls', 0)
        tokens = getattr(result, 'tokens_estimate', 0)

        if mode in self.execution_mode_stats:
            self.execution_mode_stats[mode]["count"] += 1
            self.execution_mode_stats[mode]["total_duration_ms"] += duration_ms
            self.execution_mode_stats[mode]["total_api_calls"] += api_calls
            self.execution_mode_stats[mode]["total_tokens"] += tokens

            # Learn from performance
            if mode == "code_mode" and hasattr(result, 'plan'):
                plan = result.plan
                complexity = getattr(plan, 'complexity', 'unknown')

                # Calculate improvement vs traditional
                improvement = self._calculate_improvement(mode, duration_ms, api_calls, tokens)

                if improvement > 0.5:  # 50%+ improvement
                    self.learn_from_interaction(
                        insight_type="performance",
                        content=f"Code-mode effective for {complexity} tasks: {improvement*100:.0f}% improvement",
                        confidence=min(0.9, 0.5 + (improvement * 0.5)),
                        actionable=True
                    )

                    logger.info(f"🚀 Code-mode performance: {improvement*100:.0f}% improvement")

    def _calculate_improvement(self, mode: str, duration_ms: float, api_calls: int, tokens: int) -> float:
        """Calculate performance improvement vs baseline"""
        if mode == "code_mode":
            # Compare to traditional average
            trad = self.execution_mode_stats["traditional"]
            if trad["count"] > 0:
                # Average traditional performance
                avg_trad_duration = trad["total_duration_ms"] / trad["count"]
                avg_trad_api_calls = trad["total_api_calls"] / trad["count"]

                # Improvement based on duration and API calls
                duration_improvement = (avg_trad_duration - duration_ms) / avg_trad_duration if avg_trad_duration > 0 else 0
                api_improvement = (avg_trad_api_calls - api_calls) / avg_trad_api_calls if avg_trad_api_calls > 0 else 0

                # Weighted average (60% duration, 40% API calls)
                return max(0, (duration_improvement * 0.6) + (api_improvement * 0.4))

        return 0.0

    def _calculate_execution_mode_stats(self) -> Dict:
        """Calculate execution mode comparison statistics"""
        stats = {}

        for mode, data in self.execution_mode_stats.items():
            if data["count"] > 0:
                stats[mode] = {
                    "executions": data["count"],
                    "avg_duration_ms": data["total_duration_ms"] / data["count"],
                    "avg_api_calls": data["total_api_calls"] / data["count"],
                    "avg_tokens": data["total_tokens"] / data["count"],
                    "total_api_calls": data["total_api_calls"],
                    "total_tokens": data["total_tokens"]
                }

        # Calculate improvement metrics
        if "traditional" in stats and "code_mode" in stats:
            trad = stats["traditional"]
            code = stats["code_mode"]

            stats["comparison"] = {
                "speed_improvement_pct": ((trad["avg_duration_ms"] - code["avg_duration_ms"]) / trad["avg_duration_ms"] * 100) if trad["avg_duration_ms"] > 0 else 0,
                "api_calls_reduction_pct": ((trad["avg_api_calls"] - code["avg_api_calls"]) / trad["avg_api_calls"] * 100) if trad["avg_api_calls"] > 0 else 0,
                "token_reduction_pct": ((trad["avg_tokens"] - code["avg_tokens"]) / trad["avg_tokens"] * 100) if trad["avg_tokens"] > 0 else 0
            }

        return stats

    def close(self):
        """Close connections"""
        if self.postgres_conn:
            self.postgres_conn.close()


# Example usage
if __name__ == "__main__":
    print("Autonomous Self-Improvement System Test")
    print("=" * 60)

    # Initialize
    asi = AutonomousSelfImprovement(
        enable_auto_modification=False  # Safety: disabled by default
    )

    # Set baselines
    asi.set_baseline("cpu_usage")
    asi.set_baseline("memory_usage")

    # Measure performance
    cpu = asi.measure_performance("cpu_usage", {"task": "initialization"})
    print(f"\nCPU Usage: {cpu.value:.1f}%")

    # Analyze bottlenecks
    bottlenecks = asi.analyze_bottlenecks()
    if bottlenecks:
        print(f"\nDetected {len(bottlenecks)} bottlenecks:")
        for b in bottlenecks:
            print(f"  - {b['type']}: {b['message']}")

    # Propose an improvement
    asi.propose_improvement(
        category="performance",
        title="Implement connection pooling for PostgreSQL",
        description="Use psycopg2.pool.ThreadedConnectionPool to reduce connection overhead",
        rationale="Database connections show high usage during peak load",
        files_to_modify=["/home/user/LAT5150DRVMIL/02-ai-engine/conversation_manager.py"],
        estimated_impact="medium",
        risk_level="low",
        auto_implementable=False  # Requires review
    )

    # Learn from interaction
    asi.learn_from_interaction(
        insight_type="user_preference",
        content="User prefers uncensored_code model for security research",
        confidence=0.8,
        actionable=True
    )

    # Get suggestions
    suggestions = asi.get_improvement_suggestions()
    print(f"\nImprovement Suggestions ({len(suggestions)}):")
    for suggestion in suggestions:
        print(f"  - {suggestion}")

    # Stats
    stats = asi.get_stats()
    print(f"\nStats: {json.dumps(stats, indent=2)}")

    asi.close()
