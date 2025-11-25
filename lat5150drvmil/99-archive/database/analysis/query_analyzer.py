#!/usr/bin/env python3
"""
DSMIL Token Testing Query and Analysis Engine
Advanced pattern detection, correlation analysis, and reporting tools
Version: 1.0.0
Date: 2025-09-01
"""

import sys
import sqlite3
import json
import time
import statistics
import re
from typing import Dict, List, Optional, Any, Tuple, Union
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from collections import defaultdict, Counter
import logging

# Add database backend to path
sys.path.insert(0, '/home/john/LAT5150DRVMIL/database/backends')
from database_backend import DatabaseBackend

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class AnalysisResult:
    """Analysis result structure"""
    analysis_type: str
    timestamp: float
    session_ids: List[str]
    summary: Dict[str, Any]
    details: Dict[str, Any]
    recommendations: List[str]

@dataclass
class PatternMatch:
    """Pattern match structure"""
    pattern_type: str
    pattern_name: str
    confidence: float
    tokens: List[int]
    sessions: List[str]
    evidence: Dict[str, Any]
    description: str

class QueryAnalyzer:
    """Advanced query and analysis engine for DSMIL token testing data"""
    
    def __init__(self, db_backend: DatabaseBackend):
        self.db = db_backend
        self.analysis_cache = {}
        
    def analyze_session(self, session_id: str) -> AnalysisResult:
        """Comprehensive session analysis"""
        with self.db._get_sqlite_connection() as conn:
            # Get session info
            session_info = conn.execute("""
                SELECT * FROM session_summary WHERE session_id = ?
            """, (session_id,)).fetchone()
            
            if not session_info:
                raise ValueError(f"Session {session_id} not found")
                
            session_info = dict(session_info)
            
            # Get detailed test results
            test_results = conn.execute("""
                SELECT 
                    tt.*,
                    td.hex_id,
                    td.group_id,
                    td.device_id,
                    td.potential_function
                FROM token_tests tt
                JOIN token_definitions td ON tt.token_id = td.token_id
                WHERE tt.session_id = ?
                ORDER BY tt.test_timestamp
            """, (session_id,)).fetchall()
            
            # Get thermal data
            thermal_data = conn.execute("""
                SELECT * FROM thermal_readings
                WHERE session_id = ?
                ORDER BY reading_timestamp
            """, (session_id,)).fetchall()
            
            # Get system metrics
            system_metrics = conn.execute("""
                SELECT * FROM system_metrics
                WHERE session_id = ?
                ORDER BY metric_timestamp
            """, (session_id,)).fetchall()
            
            # Analyze patterns
            patterns = self._detect_session_patterns(session_id, test_results, thermal_data, system_metrics)
            
            # Generate summary
            summary = {
                'session_info': session_info,
                'total_tests': len(test_results),
                'success_rate': session_info.get('success_rate', 0),
                'duration_hours': self._calculate_session_duration(session_info),
                'thermal_events': self._count_thermal_events(thermal_data),
                'performance_impact': self._analyze_performance_impact(system_metrics),
                'pattern_count': len(patterns)
            }
            
            # Generate details
            details = {
                'test_results': [dict(row) for row in test_results],
                'thermal_analysis': self._analyze_thermal_data(thermal_data),
                'system_performance': self._analyze_system_performance(system_metrics),
                'patterns': [asdict(p) for p in patterns],
                'token_distribution': self._analyze_token_distribution(test_results),
                'group_analysis': self._analyze_group_performance(test_results)
            }
            
            # Generate recommendations
            recommendations = self._generate_session_recommendations(summary, patterns)
            
            return AnalysisResult(
                analysis_type="session_analysis",
                timestamp=time.time(),
                session_ids=[session_id],
                summary=summary,
                details=details,
                recommendations=recommendations
            )
            
    def analyze_token_performance(self, token_id: Optional[int] = None, 
                                 hex_id: Optional[str] = None) -> AnalysisResult:
        """Analyze performance of specific token or all tokens"""
        with self.db._get_sqlite_connection() as conn:
            if token_id:
                condition = "WHERE td.token_id = ?"
                params = [token_id]
            elif hex_id:
                condition = "WHERE td.hex_id = ?"
                params = [hex_id]
            else:
                condition = ""
                params = []
                
            # Get token performance data
            token_data = conn.execute(f"""
                SELECT 
                    td.*,
                    COUNT(tt.test_id) as total_tests,
                    SUM(CASE WHEN tt.success THEN 1 ELSE 0 END) as successful_tests,
                    AVG(tt.test_duration_ms) as avg_duration_ms,
                    MIN(tt.test_timestamp) as first_test,
                    MAX(tt.test_timestamp) as last_test,
                    COUNT(DISTINCT tt.session_id) as session_count
                FROM token_definitions td
                LEFT JOIN token_tests tt ON td.token_id = tt.token_id
                {condition}
                GROUP BY td.token_id
                HAVING COUNT(tt.test_id) > 0
                ORDER BY successful_tests DESC, total_tests DESC
            """, params).fetchall()
            
            if not token_data:
                raise ValueError("No test data found for specified token(s)")
                
            # Analyze success patterns
            success_patterns = self._analyze_token_success_patterns(token_data)
            
            # Analyze correlations
            correlations = self._find_token_correlations(token_data)
            
            summary = {
                'tokens_analyzed': len(token_data),
                'total_tests': sum(row['total_tests'] for row in token_data),
                'overall_success_rate': self._calculate_overall_success_rate(token_data),
                'avg_test_duration': statistics.mean([row['avg_duration_ms'] for row in token_data if row['avg_duration_ms']]),
                'pattern_matches': len(success_patterns),
                'correlation_count': len(correlations)
            }
            
            details = {
                'token_performance': [dict(row) for row in token_data],
                'success_patterns': [asdict(p) for p in success_patterns],
                'correlations': correlations,
                'group_breakdown': self._breakdown_by_group(token_data),
                'function_breakdown': self._breakdown_by_function(token_data)
            }
            
            recommendations = self._generate_token_recommendations(token_data, success_patterns)
            
            return AnalysisResult(
                analysis_type="token_performance",
                timestamp=time.time(),
                session_ids=self._extract_session_ids_from_tokens(token_data),
                summary=summary,
                details=details,
                recommendations=recommendations
            )
            
    def detect_thermal_correlations(self, session_ids: Optional[List[str]] = None) -> AnalysisResult:
        """Detect correlations between token operations and thermal events"""
        with self.db._get_sqlite_connection() as conn:
            if session_ids:
                session_condition = "AND tt.session_id IN ({})".format(','.join(['?' for _ in session_ids]))
                params = session_ids
            else:
                session_condition = ""
                params = []
                
            # Get token tests with thermal data
            thermal_correlations = conn.execute(f"""
                SELECT 
                    tt.test_id,
                    tt.session_id,
                    tt.token_id,
                    td.hex_id,
                    td.group_id,
                    td.potential_function,
                    tt.test_timestamp,
                    tr_before.temperature_celsius as temp_before,
                    tr_after.temperature_celsius as temp_after,
                    tr_after.temperature_celsius - tr_before.temperature_celsius as temp_delta,
                    tr_after.thermal_state
                FROM token_tests tt
                JOIN token_definitions td ON tt.token_id = td.token_id
                LEFT JOIN thermal_readings tr_before ON tt.session_id = tr_before.session_id
                    AND tr_before.reading_timestamp <= tt.test_timestamp
                    AND tr_before.reading_timestamp >= tt.test_timestamp - (30.0 / 86400.0)  -- 30 seconds before
                LEFT JOIN thermal_readings tr_after ON tt.session_id = tr_after.session_id
                    AND tr_after.reading_timestamp >= tt.test_timestamp
                    AND tr_after.reading_timestamp <= tt.test_timestamp + (60.0 / 86400.0)  -- 60 seconds after
                WHERE tr_before.reading_id IS NOT NULL 
                    AND tr_after.reading_id IS NOT NULL
                    {session_condition}
                ORDER BY ABS(tr_after.temperature_celsius - tr_before.temperature_celsius) DESC
            """, params).fetchall()
            
            # Analyze thermal impact patterns
            impact_patterns = self._analyze_thermal_impact_patterns(thermal_correlations)
            
            # Find high-impact tokens
            high_impact_tokens = self._identify_high_thermal_impact_tokens(thermal_correlations)
            
            summary = {
                'correlations_found': len(thermal_correlations),
                'high_impact_tokens': len(high_impact_tokens),
                'max_temp_delta': max([row['temp_delta'] for row in thermal_correlations if row['temp_delta']], default=0),
                'thermal_events': len([row for row in thermal_correlations if row['thermal_state'] != 'normal']),
                'pattern_count': len(impact_patterns)
            }
            
            details = {
                'thermal_correlations': [dict(row) for row in thermal_correlations],
                'impact_patterns': [asdict(p) for p in impact_patterns],
                'high_impact_tokens': high_impact_tokens,
                'temperature_statistics': self._calculate_temp_statistics(thermal_correlations),
                'group_thermal_impact': self._analyze_group_thermal_impact(thermal_correlations)
            }
            
            recommendations = self._generate_thermal_recommendations(thermal_correlations, impact_patterns)
            
            return AnalysisResult(
                analysis_type="thermal_correlation",
                timestamp=time.time(),
                session_ids=session_ids or self._extract_session_ids_from_correlations(thermal_correlations),
                summary=summary,
                details=details,
                recommendations=recommendations
            )
            
    def find_access_patterns(self, session_ids: Optional[List[str]] = None) -> AnalysisResult:
        """Find patterns in token access methods and success rates"""
        with self.db._get_sqlite_connection() as conn:
            if session_ids:
                condition = "WHERE tt.session_id IN ({})".format(','.join(['?' for _ in session_ids]))
                params = session_ids
            else:
                condition = ""
                params = []
                
            # Analyze access method effectiveness
            access_analysis = conn.execute(f"""
                SELECT 
                    tt.access_method,
                    tt.operation_type,
                    td.group_id,
                    COUNT(*) as total_attempts,
                    SUM(CASE WHEN tt.success THEN 1 ELSE 0 END) as successful_attempts,
                    AVG(tt.test_duration_ms) as avg_duration,
                    COUNT(DISTINCT tt.token_id) as unique_tokens,
                    COUNT(DISTINCT tt.session_id) as session_count
                FROM token_tests tt
                JOIN token_definitions td ON tt.token_id = td.token_id
                {condition}
                GROUP BY tt.access_method, tt.operation_type, td.group_id
                ORDER BY successful_attempts DESC, total_attempts DESC
            """, params).fetchall()
            
            # Find optimal access patterns
            optimal_patterns = self._find_optimal_access_patterns(access_analysis)
            
            # Analyze failure patterns
            failure_patterns = self._analyze_failure_patterns(session_ids, params)
            
            summary = {
                'access_combinations': len(access_analysis),
                'optimal_patterns': len(optimal_patterns),
                'failure_patterns': len(failure_patterns),
                'best_success_rate': max([self._calculate_success_rate(row) for row in access_analysis], default=0),
                'total_attempts': sum(row['total_attempts'] for row in access_analysis)
            }
            
            details = {
                'access_analysis': [dict(row) for row in access_analysis],
                'optimal_patterns': [asdict(p) for p in optimal_patterns],
                'failure_patterns': [asdict(p) for p in failure_patterns],
                'method_comparison': self._compare_access_methods(access_analysis),
                'operation_effectiveness': self._analyze_operation_effectiveness(access_analysis)
            }
            
            recommendations = self._generate_access_recommendations(optimal_patterns, failure_patterns)
            
            return AnalysisResult(
                analysis_type="access_patterns",
                timestamp=time.time(),
                session_ids=session_ids or [],
                summary=summary,
                details=details,
                recommendations=recommendations
            )
            
    def generate_comprehensive_report(self, session_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """Generate a comprehensive analysis report"""
        report = {
            'generated_at': time.time(),
            'report_type': 'comprehensive',
            'session_scope': session_ids or 'all_sessions',
            'analyses': {}
        }
        
        try:
            # Session analysis (if specific sessions provided)
            if session_ids:
                session_analyses = []
                for session_id in session_ids:
                    try:
                        analysis = self.analyze_session(session_id)
                        session_analyses.append(asdict(analysis))
                    except Exception as e:
                        logger.error(f"Failed to analyze session {session_id}: {str(e)}")
                report['analyses']['sessions'] = session_analyses
                
            # Token performance analysis
            try:
                token_analysis = self.analyze_token_performance()
                report['analyses']['token_performance'] = asdict(token_analysis)
            except Exception as e:
                logger.error(f"Failed to analyze token performance: {str(e)}")
                
            # Thermal correlation analysis
            try:
                thermal_analysis = self.detect_thermal_correlations(session_ids)
                report['analyses']['thermal_correlations'] = asdict(thermal_analysis)
            except Exception as e:
                logger.error(f"Failed to analyze thermal correlations: {str(e)}")
                
            # Access pattern analysis
            try:
                access_analysis = self.find_access_patterns(session_ids)
                report['analyses']['access_patterns'] = asdict(access_analysis)
            except Exception as e:
                logger.error(f"Failed to analyze access patterns: {str(e)}")
                
            # Overall system health
            try:
                system_health = self._analyze_overall_system_health(session_ids)
                report['analyses']['system_health'] = system_health
            except Exception as e:
                logger.error(f"Failed to analyze system health: {str(e)}")
                
        except Exception as e:
            logger.error(f"Failed to generate comprehensive report: {str(e)}")
            
        return report
        
    # Pattern detection methods
    def _detect_session_patterns(self, session_id: str, test_results: List, 
                                thermal_data: List, system_metrics: List) -> List[PatternMatch]:
        """Detect patterns within a session"""
        patterns = []
        
        # Sequential success pattern
        if len(test_results) > 3:
            sequential_successes = self._find_sequential_patterns(test_results)
            if sequential_successes:
                patterns.append(PatternMatch(
                    pattern_type="sequential",
                    pattern_name="sequential_success",
                    confidence=0.8,
                    tokens=[t['token_id'] for t in sequential_successes],
                    sessions=[session_id],
                    evidence={"sequence_length": len(sequential_successes)},
                    description=f"Found sequential success pattern with {len(sequential_successes)} tokens"
                ))
                
        # Thermal threshold pattern
        thermal_events = [t for t in thermal_data if t['thermal_state'] != 'normal']
        if thermal_events:
            patterns.append(PatternMatch(
                pattern_type="thermal",
                pattern_name="thermal_events",
                confidence=0.9,
                tokens=list(set(t['token_id'] for t in test_results if t['success'] and any(
                    abs(t['test_timestamp'] - th['reading_timestamp']) < 30 for th in thermal_events
                ))),
                sessions=[session_id],
                evidence={"event_count": len(thermal_events)},
                description=f"Found {len(thermal_events)} thermal events during testing"
            ))
            
        # Group activation pattern
        group_tests = defaultdict(list)
        for test in test_results:
            group_tests[test['group_id']].append(test)
            
        for group_id, tests in group_tests.items():
            if len(tests) > 6 and all(t['success'] for t in tests[-6:]):  # Last 6 tests in group successful
                patterns.append(PatternMatch(
                    pattern_type="group",
                    pattern_name="group_activation",
                    confidence=0.7,
                    tokens=[t['token_id'] for t in tests[-6:]],
                    sessions=[session_id],
                    evidence={"group_id": group_id, "success_streak": 6},
                    description=f"Group {group_id} shows activation pattern with 6+ consecutive successes"
                ))
                
        return patterns
        
    def _find_sequential_patterns(self, test_results: List) -> List:
        """Find sequential success patterns"""
        sequential = []
        current_streak = []
        
        for test in test_results:
            if test['success']:
                current_streak.append(test)
            else:
                if len(current_streak) >= 3:
                    sequential.extend(current_streak)
                current_streak = []
                
        if len(current_streak) >= 3:
            sequential.extend(current_streak)
            
        return sequential
        
    def _analyze_thermal_impact_patterns(self, thermal_correlations: List) -> List[PatternMatch]:
        """Analyze thermal impact patterns"""
        patterns = []
        
        # High impact tokens
        high_impact = [tc for tc in thermal_correlations if tc['temp_delta'] and tc['temp_delta'] > 2.0]
        if high_impact:
            token_impacts = defaultdict(list)
            for tc in high_impact:
                token_impacts[tc['token_id']].append(tc['temp_delta'])
                
            for token_id, deltas in token_impacts.items():
                if len(deltas) >= 2 and statistics.mean(deltas) > 3.0:
                    patterns.append(PatternMatch(
                        pattern_type="thermal_impact",
                        pattern_name="high_thermal_impact",
                        confidence=0.9,
                        tokens=[token_id],
                        sessions=list(set(tc['session_id'] for tc in high_impact if tc['token_id'] == token_id)),
                        evidence={"avg_temp_delta": statistics.mean(deltas), "occurrence_count": len(deltas)},
                        description=f"Token {token_id} consistently causes high thermal impact (avg +{statistics.mean(deltas):.1f}°C)"
                    ))
                    
        return patterns
        
    def _find_optimal_access_patterns(self, access_analysis: List) -> List[PatternMatch]:
        """Find optimal access method patterns"""
        patterns = []
        
        # Group by method and find best performers
        method_performance = defaultdict(list)
        for row in access_analysis:
            success_rate = self._calculate_success_rate(row)
            method_performance[row['access_method']].append(success_rate)
            
        for method, success_rates in method_performance.items():
            if len(success_rates) > 1 and statistics.mean(success_rates) > 0.8:
                patterns.append(PatternMatch(
                    pattern_type="access_method",
                    pattern_name="optimal_method",
                    confidence=0.8,
                    tokens=[],
                    sessions=[],
                    evidence={"method": method, "avg_success_rate": statistics.mean(success_rates)},
                    description=f"Access method '{method}' shows consistently high success rate ({statistics.mean(success_rates):.1%})"
                ))
                
        return patterns
        
    # Helper methods for calculations and analysis
    def _calculate_session_duration(self, session_info: Dict) -> float:
        """Calculate session duration in hours"""
        start = session_info.get('start_time')
        end = session_info.get('end_time')
        if start and end:
            try:
                start_dt = datetime.fromisoformat(start)
                end_dt = datetime.fromisoformat(end)
                return (end_dt - start_dt).total_seconds() / 3600
            except:
                pass
        return 0.0
        
    def _count_thermal_events(self, thermal_data: List) -> int:
        """Count thermal events (non-normal states)"""
        return len([t for t in thermal_data if t['thermal_state'] != 'normal'])
        
    def _analyze_performance_impact(self, system_metrics: List) -> Dict:
        """Analyze system performance impact"""
        if not system_metrics:
            return {}
            
        cpu_values = [m['cpu_percent'] for m in system_metrics if m['cpu_percent'] is not None]
        memory_values = [m['memory_percent'] for m in system_metrics if m['memory_percent'] is not None]
        
        return {
            'avg_cpu_usage': statistics.mean(cpu_values) if cpu_values else 0,
            'max_cpu_usage': max(cpu_values) if cpu_values else 0,
            'avg_memory_usage': statistics.mean(memory_values) if memory_values else 0,
            'max_memory_usage': max(memory_values) if memory_values else 0,
            'high_cpu_events': len([c for c in cpu_values if c > 80]),
            'high_memory_events': len([m for m in memory_values if m > 85])
        }
        
    def _calculate_success_rate(self, row: Dict) -> float:
        """Calculate success rate from analysis row"""
        total = row.get('total_attempts', 0)
        successful = row.get('successful_attempts', 0)
        return successful / total if total > 0 else 0.0
        
    def _generate_session_recommendations(self, summary: Dict, patterns: List[PatternMatch]) -> List[str]:
        """Generate recommendations based on session analysis"""
        recommendations = []
        
        if summary.get('success_rate', 0) < 0.5:
            recommendations.append("Low success rate detected. Consider reviewing token access methods and system conditions.")
            
        if summary.get('thermal_events', 0) > 0:
            recommendations.append("Thermal events occurred during testing. Implement cooling breaks between operations.")
            
        if any(p.pattern_name == "high_thermal_impact" for p in patterns):
            recommendations.append("High thermal impact tokens identified. Schedule these for cooler system conditions.")
            
        if summary.get('pattern_count', 0) > 5:
            recommendations.append("Multiple patterns detected. Consider developing automated test sequences based on patterns.")
            
        return recommendations
        
    def _generate_token_recommendations(self, token_data: List, patterns: List[PatternMatch]) -> List[str]:
        """Generate recommendations for token operations"""
        recommendations = []
        
        # Find tokens with low success rates
        problematic_tokens = [t for t in token_data if self._calculate_success_rate(t) < 0.3]
        if problematic_tokens:
            recommendations.append(f"Found {len(problematic_tokens)} tokens with low success rates. Consider alternative access methods.")
            
        # Find high-performance tokens
        reliable_tokens = [t for t in token_data if self._calculate_success_rate(t) > 0.9 and t['total_tests'] > 5]
        if reliable_tokens:
            recommendations.append(f"Found {len(reliable_tokens)} highly reliable tokens. Use these for baseline testing.")
            
        return recommendations
        
    def _generate_thermal_recommendations(self, correlations: List, patterns: List[PatternMatch]) -> List[str]:
        """Generate thermal-related recommendations"""
        recommendations = []
        
        high_impact_count = len([c for c in correlations if c.get('temp_delta', 0) > 5.0])
        if high_impact_count > 0:
            recommendations.append(f"Found {high_impact_count} high thermal impact operations. Implement mandatory cooling periods.")
            
        if any(c.get('thermal_state') == 'critical' for c in correlations):
            recommendations.append("Critical thermal events detected. Review system thermal management and reduce test intensity.")
            
        return recommendations
        
    def _generate_access_recommendations(self, optimal_patterns: List[PatternMatch], failure_patterns: List[PatternMatch]) -> List[str]:
        """Generate access method recommendations"""
        recommendations = []
        
        if optimal_patterns:
            best_methods = [p.evidence.get('method') for p in optimal_patterns]
            recommendations.append(f"Optimal access methods identified: {', '.join(set(best_methods))}")
            
        if failure_patterns:
            recommendations.append("Failure patterns detected. Consider implementing retry logic with different access methods.")
            
        return recommendations

    # Additional helper methods
    def _analyze_thermal_data(self, thermal_data: List) -> Dict:
        """Analyze thermal readings"""
        if not thermal_data:
            return {}
            
        temps = [t['temperature_celsius'] for t in thermal_data]
        return {
            'min_temp': min(temps),
            'max_temp': max(temps),
            'avg_temp': statistics.mean(temps),
            'thermal_events': len([t for t in thermal_data if t['thermal_state'] != 'normal']),
            'sensors': list(set(t['sensor_name'] for t in thermal_data))
        }
        
    def _analyze_system_performance(self, system_metrics: List) -> Dict:
        """Analyze system performance metrics"""
        if not system_metrics:
            return {}
            
        return {
            'metric_count': len(system_metrics),
            'avg_cpu': statistics.mean([m['cpu_percent'] for m in system_metrics if m['cpu_percent']]) if any(m['cpu_percent'] for m in system_metrics) else 0,
            'avg_memory': statistics.mean([m['memory_percent'] for m in system_metrics if m['memory_percent']]) if any(m['memory_percent'] for m in system_metrics) else 0,
            'performance_events': len([m for m in system_metrics if (m.get('cpu_percent', 0) > 80 or m.get('memory_percent', 0) > 85)])
        }
        
    def _analyze_failure_patterns(self, session_ids: Optional[List[str]], params: List) -> List[PatternMatch]:
        """Analyze failure patterns in token tests"""
        patterns = []
        
        with self.db._get_sqlite_connection() as conn:
            condition = ""
            if session_ids:
                condition = "WHERE tt.session_id IN ({})".format(','.join(['?' for _ in session_ids]))
                
            failures = conn.execute(f"""
                SELECT 
                    tt.token_id,
                    tt.error_code,
                    tt.error_message,
                    td.group_id,
                    td.potential_function,
                    COUNT(*) as failure_count
                FROM token_tests tt
                JOIN token_definitions td ON tt.token_id = td.token_id
                {condition} AND tt.success = 0
                GROUP BY tt.token_id, tt.error_code
                HAVING failure_count > 2
                ORDER BY failure_count DESC
            """, params).fetchall()
            
            for failure in failures:
                patterns.append(PatternMatch(
                    pattern_type="failure",
                    pattern_name="recurring_failure",
                    confidence=0.8,
                    tokens=[failure['token_id']],
                    sessions=session_ids or [],
                    evidence={
                        "error_code": failure['error_code'],
                        "error_message": failure['error_message'],
                        "failure_count": failure['failure_count']
                    },
                    description=f"Token {failure['token_id']} shows recurring failure pattern ({failure['failure_count']} occurrences)"
                ))
                
        return patterns

if __name__ == "__main__":
    # Example usage
    from database_backend import DatabaseBackend
    
    db = DatabaseBackend()
    analyzer = QueryAnalyzer(db)
    
    # Example analysis
    try:
        # Generate comprehensive report
        report = analyzer.generate_comprehensive_report()
        print("Generated comprehensive analysis report")
        print(f"Report contains {len(report.get('analyses', {}))} analysis types")
        
        # Save report
        with open('/home/john/LAT5150DRVMIL/database/analysis/latest_report.json', 'w') as f:
            json.dump(report, f, indent=2)
            
    except Exception as e:
        print(f"Analysis failed: {str(e)}")