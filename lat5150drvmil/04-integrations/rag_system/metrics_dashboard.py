"""
Metrics & Analytics Dashboard (Phase 3.1)

Track code quality metrics over time for the LAT5150DRVMIL system.
Stores historical data and generates trend reports.

Features:
- Security vulnerability tracking (trend over time)
- Code complexity evolution (average, max, distribution)
- Test coverage tracking
- Performance metrics
- Technical debt score
- CLI visualization with sparklines
- Local storage (SQLite database)
- Historical analysis and trends

Example:
    >>> dashboard = MetricsDashboard()
    >>> dashboard.record_analysis(code, results)
    >>> dashboard.show_trends(days=30)
    >>> dashboard.generate_report()
"""

import os
import json
import sqlite3
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import hashlib


@dataclass
class CodeMetrics:
    """Metrics snapshot for a piece of code"""
    timestamp: datetime
    file_path: str
    code_hash: str

    # Security metrics
    security_critical: int = 0
    security_high: int = 0
    security_medium: int = 0
    security_low: int = 0

    # Complexity metrics
    cyclomatic_complexity: int = 0
    cognitive_complexity: int = 0
    nesting_depth: int = 0
    num_functions: int = 0
    num_classes: int = 0

    # Performance metrics
    perf_issues: int = 0
    perf_score: float = 10.0

    # Code quality
    code_smells: int = 0
    maintainability_score: float = 10.0

    # Test coverage (if available)
    test_coverage: Optional[float] = None

    # Size metrics
    lines_of_code: int = 0
    comment_ratio: float = 0.0

    # Technical debt
    technical_debt_score: float = 0.0  # Higher = more debt


@dataclass
class TrendAnalysis:
    """Analysis of metric trends over time"""
    metric_name: str
    current_value: float
    previous_value: float
    change_percent: float
    trend: str  # "improving", "degrading", "stable"
    sparkline: str
    historical_values: List[float] = field(default_factory=list)


class MetricsDatabase:
    """SQLite database for storing metrics"""

    def __init__(self, db_path: str = ".code_metrics.db"):
        self.db_path = db_path
        self.conn = None
        self._init_database()

    def _init_database(self):
        """Initialize database schema"""
        self.conn = sqlite3.connect(self.db_path)
        cursor = self.conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                file_path TEXT NOT NULL,
                code_hash TEXT NOT NULL,

                -- Security
                security_critical INTEGER DEFAULT 0,
                security_high INTEGER DEFAULT 0,
                security_medium INTEGER DEFAULT 0,
                security_low INTEGER DEFAULT 0,

                -- Complexity
                cyclomatic_complexity INTEGER DEFAULT 0,
                cognitive_complexity INTEGER DEFAULT 0,
                nesting_depth INTEGER DEFAULT 0,
                num_functions INTEGER DEFAULT 0,
                num_classes INTEGER DEFAULT 0,

                -- Performance
                perf_issues INTEGER DEFAULT 0,
                perf_score REAL DEFAULT 10.0,

                -- Quality
                code_smells INTEGER DEFAULT 0,
                maintainability_score REAL DEFAULT 10.0,

                -- Coverage
                test_coverage REAL,

                -- Size
                lines_of_code INTEGER DEFAULT 0,
                comment_ratio REAL DEFAULT 0.0,

                -- Technical debt
                technical_debt_score REAL DEFAULT 0.0
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_timestamp
            ON metrics(timestamp)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_file_path
            ON metrics(file_path)
        """)

        self.conn.commit()

    def insert_metrics(self, metrics: CodeMetrics):
        """Insert metrics into database"""
        cursor = self.conn.cursor()

        cursor.execute("""
            INSERT INTO metrics (
                timestamp, file_path, code_hash,
                security_critical, security_high, security_medium, security_low,
                cyclomatic_complexity, cognitive_complexity, nesting_depth,
                num_functions, num_classes,
                perf_issues, perf_score,
                code_smells, maintainability_score,
                test_coverage,
                lines_of_code, comment_ratio,
                technical_debt_score
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            metrics.timestamp.isoformat(),
            metrics.file_path,
            metrics.code_hash,
            metrics.security_critical,
            metrics.security_high,
            metrics.security_medium,
            metrics.security_low,
            metrics.cyclomatic_complexity,
            metrics.cognitive_complexity,
            metrics.nesting_depth,
            metrics.num_functions,
            metrics.num_classes,
            metrics.perf_issues,
            metrics.perf_score,
            metrics.code_smells,
            metrics.maintainability_score,
            metrics.test_coverage,
            metrics.lines_of_code,
            metrics.comment_ratio,
            metrics.technical_debt_score
        ))

        self.conn.commit()

    def get_recent_metrics(self, days: int = 30, file_path: Optional[str] = None) -> List[CodeMetrics]:
        """Get metrics from last N days"""
        cursor = self.conn.cursor()
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()

        if file_path:
            cursor.execute("""
                SELECT * FROM metrics
                WHERE timestamp >= ? AND file_path = ?
                ORDER BY timestamp DESC
            """, (cutoff, file_path))
        else:
            cursor.execute("""
                SELECT * FROM metrics
                WHERE timestamp >= ?
                ORDER BY timestamp DESC
            """, (cutoff,))

        rows = cursor.fetchall()
        return [self._row_to_metrics(row) for row in rows]

    def get_metric_history(self, metric_name: str, days: int = 30) -> List[Tuple[datetime, float]]:
        """Get history of a specific metric"""
        cursor = self.conn.cursor()
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()

        # Map metric name to column
        valid_metrics = {
            'security_total': 'security_critical + security_high + security_medium + security_low',
            'cyclomatic_complexity': 'AVG(cyclomatic_complexity)',
            'maintainability_score': 'AVG(maintainability_score)',
            'technical_debt': 'AVG(technical_debt_score)',
            'code_smells': 'SUM(code_smells)',
        }

        if metric_name not in valid_metrics:
            return []

        query = f"""
            SELECT timestamp, {valid_metrics[metric_name]} as value
            FROM metrics
            WHERE timestamp >= ?
            GROUP BY DATE(timestamp)
            ORDER BY timestamp ASC
        """

        cursor.execute(query, (cutoff,))
        rows = cursor.fetchall()

        return [(datetime.fromisoformat(row[0]), float(row[1])) for row in rows]

    def _row_to_metrics(self, row) -> CodeMetrics:
        """Convert database row to CodeMetrics"""
        return CodeMetrics(
            timestamp=datetime.fromisoformat(row[1]),
            file_path=row[2],
            code_hash=row[3],
            security_critical=row[4],
            security_high=row[5],
            security_medium=row[6],
            security_low=row[7],
            cyclomatic_complexity=row[8],
            cognitive_complexity=row[9],
            nesting_depth=row[10],
            num_functions=row[11],
            num_classes=row[12],
            perf_issues=row[13],
            perf_score=row[14],
            code_smells=row[15],
            maintainability_score=row[16],
            test_coverage=row[17],
            lines_of_code=row[18],
            comment_ratio=row[19],
            technical_debt_score=row[20]
        )

    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()


class SparklineGenerator:
    """Generate ASCII sparklines for metric trends"""

    CHARS = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█']

    @classmethod
    def generate(cls, values: List[float], width: int = 20) -> str:
        """Generate sparkline from values"""
        if not values:
            return ""

        # Normalize values to 0-7 range
        min_val = min(values)
        max_val = max(values)

        if max_val == min_val:
            return cls.CHARS[3] * min(len(values), width)

        normalized = [
            int(((v - min_val) / (max_val - min_val)) * 7)
            for v in values
        ]

        # Downsample if too many values
        if len(normalized) > width:
            step = len(normalized) / width
            normalized = [normalized[int(i * step)] for i in range(width)]

        return ''.join(cls.CHARS[n] for n in normalized)


class MetricsDashboard:
    """Main metrics dashboard"""

    def __init__(self, db_path: str = ".code_metrics.db"):
        self.db = MetricsDatabase(db_path)

    def record_analysis(self, code: str, file_path: str, analysis_results: Dict):
        """Record analysis results as metrics"""

        # Calculate code hash
        code_hash = hashlib.md5(code.encode()).hexdigest()

        # Extract metrics from analysis results
        metrics = CodeMetrics(
            timestamp=datetime.now(),
            file_path=file_path,
            code_hash=code_hash
        )

        # Security metrics
        if 'security' in analysis_results:
            sec = analysis_results['security']
            for issues in sec.values():
                for issue in issues:
                    severity = issue.get('severity', 'low')
                    if severity == 'critical':
                        metrics.security_critical += 1
                    elif severity == 'high':
                        metrics.security_high += 1
                    elif severity == 'medium':
                        metrics.security_medium += 1
                    else:
                        metrics.security_low += 1

        # Complexity metrics
        if 'complexity' in analysis_results:
            comp = analysis_results['complexity']
            metrics.cyclomatic_complexity = comp.get('cyclomatic_complexity', 0)
            metrics.nesting_depth = comp.get('nesting_depth', 0)
            metrics.num_functions = comp.get('num_functions', 0)
            metrics.num_classes = comp.get('num_classes', 0)

        # Performance metrics
        if 'performance' in analysis_results:
            perf = analysis_results['performance']
            metrics.perf_issues = len(perf.get('issues', []))
            metrics.perf_score = perf.get('score', 10.0)

        # Code smells
        if 'smells' in analysis_results:
            metrics.code_smells = len(analysis_results['smells'])

        # Code size
        metrics.lines_of_code = len(code.split('\n'))
        comment_lines = len([line for line in code.split('\n') if line.strip().startswith('#')])
        metrics.comment_ratio = comment_lines / max(metrics.lines_of_code, 1)

        # Calculate technical debt score
        metrics.technical_debt_score = self._calculate_technical_debt(metrics)

        # Store in database
        self.db.insert_metrics(metrics)

        return metrics

    def _calculate_technical_debt(self, metrics: CodeMetrics) -> float:
        """Calculate technical debt score (0-100, higher = more debt)"""
        debt = 0.0

        # Security debt (critical issues are expensive)
        debt += metrics.security_critical * 10.0
        debt += metrics.security_high * 5.0
        debt += metrics.security_medium * 2.0
        debt += metrics.security_low * 0.5

        # Complexity debt
        if metrics.cyclomatic_complexity > 10:
            debt += (metrics.cyclomatic_complexity - 10) * 1.5

        if metrics.nesting_depth > 4:
            debt += (metrics.nesting_depth - 4) * 2.0

        # Code smell debt
        debt += metrics.code_smells * 1.0

        # Performance debt
        debt += metrics.perf_issues * 1.5

        # Lack of documentation debt
        if metrics.comment_ratio < 0.1:  # Less than 10% comments
            debt += 5.0

        return min(debt, 100.0)  # Cap at 100

    def analyze_trends(self, days: int = 30) -> List[TrendAnalysis]:
        """Analyze metric trends over time"""
        trends = []

        # Key metrics to track
        metrics_to_track = [
            'security_total',
            'cyclomatic_complexity',
            'maintainability_score',
            'technical_debt',
            'code_smells'
        ]

        for metric_name in metrics_to_track:
            history = self.db.get_metric_history(metric_name, days)

            if len(history) < 2:
                continue

            # Get current and previous values
            values = [v for _, v in history]
            current = values[-1]
            previous = values[0]

            # Calculate change
            if previous != 0:
                change_percent = ((current - previous) / previous) * 100
            else:
                change_percent = 0.0

            # Determine trend direction
            if abs(change_percent) < 5:
                trend = "stable"
            elif metric_name in ['maintainability_score']:
                # Higher is better
                trend = "improving" if change_percent > 0 else "degrading"
            else:
                # Lower is better
                trend = "improving" if change_percent < 0 else "degrading"

            # Generate sparkline
            sparkline = SparklineGenerator.generate(values, width=30)

            trends.append(TrendAnalysis(
                metric_name=metric_name,
                current_value=current,
                previous_value=previous,
                change_percent=change_percent,
                trend=trend,
                sparkline=sparkline,
                historical_values=values
            ))

        return trends

    def show_trends(self, days: int = 30):
        """Display metric trends in CLI"""
        trends = self.analyze_trends(days)

        print("\n" + "=" * 80)
        print(f"📊 CODE QUALITY TRENDS (Last {days} days)")
        print("=" * 80)

        for trend in trends:
            # Format metric name
            name = trend.metric_name.replace('_', ' ').title()

            # Trend indicator
            if trend.trend == "improving":
                indicator = "📈 ↗"
                color = "\033[92m"  # Green
            elif trend.trend == "degrading":
                indicator = "📉 ↘"
                color = "\033[91m"  # Red
            else:
                indicator = "📊 →"
                color = "\033[93m"  # Yellow

            reset = "\033[0m"

            print(f"\n{name}:")
            print(f"  Current: {trend.current_value:.1f}")
            print(f"  Change: {color}{trend.change_percent:+.1f}%{reset} {indicator}")
            print(f"  Trend: {trend.sparkline}")

        print("\n" + "=" * 80)

    def generate_report(self, days: int = 30, output_file: Optional[str] = None):
        """Generate comprehensive metrics report"""

        recent_metrics = self.db.get_recent_metrics(days)
        trends = self.analyze_trends(days)

        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append(f"CODE QUALITY REPORT - LAT5150DRVMIL")
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Period: Last {days} days")
        report_lines.append("=" * 80)
        report_lines.append("")

        # Summary statistics
        if recent_metrics:
            latest = recent_metrics[0]

            report_lines.append("CURRENT STATUS:")
            report_lines.append("-" * 80)
            report_lines.append(f"  Security Issues: {latest.security_critical + latest.security_high + latest.security_medium + latest.security_low}")
            report_lines.append(f"    Critical: {latest.security_critical}")
            report_lines.append(f"    High: {latest.security_high}")
            report_lines.append(f"    Medium: {latest.security_medium}")
            report_lines.append(f"    Low: {latest.security_low}")
            report_lines.append("")
            report_lines.append(f"  Complexity:")
            report_lines.append(f"    Cyclomatic: {latest.cyclomatic_complexity}")
            report_lines.append(f"    Nesting Depth: {latest.nesting_depth}")
            report_lines.append("")
            report_lines.append(f"  Code Quality:")
            report_lines.append(f"    Maintainability Score: {latest.maintainability_score:.1f}/10")
            report_lines.append(f"    Code Smells: {latest.code_smells}")
            report_lines.append(f"    Technical Debt: {latest.technical_debt_score:.1f}")
            report_lines.append("")
            report_lines.append(f"  Metrics:")
            report_lines.append(f"    Lines of Code: {latest.lines_of_code}")
            report_lines.append(f"    Comment Ratio: {latest.comment_ratio:.1%}")
            report_lines.append("")

        # Trends
        report_lines.append("TRENDS:")
        report_lines.append("-" * 80)
        for trend in trends:
            name = trend.metric_name.replace('_', ' ').title()
            indicator = "↗" if trend.trend == "improving" else "↘" if trend.trend == "degrading" else "→"
            report_lines.append(f"  {name}: {trend.current_value:.1f} ({trend.change_percent:+.1f}%) {indicator}")
            report_lines.append(f"    {trend.sparkline}")

        report_lines.append("")
        report_lines.append("=" * 80)

        report_text = "\n".join(report_lines)

        # Print to console
        print(report_text)

        # Save to file if requested
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report_text)
            print(f"\n✓ Report saved to {output_file}")

        return report_text

    def get_summary(self) -> Dict:
        """Get summary statistics"""
        recent = self.db.get_recent_metrics(days=1)

        if not recent:
            return {}

        latest = recent[0]

        return {
            'timestamp': latest.timestamp.isoformat(),
            'security_total': latest.security_critical + latest.security_high + latest.security_medium + latest.security_low,
            'complexity': latest.cyclomatic_complexity,
            'maintainability': latest.maintainability_score,
            'technical_debt': latest.technical_debt_score,
            'lines_of_code': latest.lines_of_code
        }

    def close(self):
        """Close database connection"""
        self.db.close()


# Example usage
if __name__ == "__main__":
    # Create dashboard
    dashboard = MetricsDashboard()

    # Example: Record some metrics
    test_code = '''
def process_data(x, y, z):
    """Process data with security issue"""
    query = "SELECT * FROM users WHERE id = %s" % user_id  # SQL injection

    for i in range(100):
        for j in range(100):
            result = i * j

    return result
'''

    # Simulate analysis results
    analysis_results = {
        'security': {
            'sql_injection': [{'severity': 'high', 'line': 3}]
        },
        'complexity': {
            'cyclomatic_complexity': 15,
            'nesting_depth': 3,
            'num_functions': 1,
            'num_classes': 0
        },
        'performance': {
            'issues': [{'type': 'nested_loop'}],
            'score': 7.0
        },
        'smells': [{'type': 'long_function'}]
    }

    dashboard.record_analysis(test_code, "test.py", analysis_results)

    # Show trends
    dashboard.show_trends(days=30)

    # Generate report
    dashboard.generate_report(days=30)

    dashboard.close()
