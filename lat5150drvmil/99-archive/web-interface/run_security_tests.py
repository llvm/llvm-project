#!/usr/bin/env python3
"""
DSMIL Phase 3 Security Testing Execution Script
Comprehensive security testing launcher with dependency management

Classification: RESTRICTED
Usage: python3 run_security_tests.py [options]
"""

import sys
import subprocess
import importlib
import asyncio
import argparse
import logging
import os
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("SecurityTestLauncher")

def check_and_install_dependencies():
    """Check and install required dependencies"""
    required_packages = [
        'aiohttp',
        'asyncio',
        'psutil',
        'jwt',
        'passlib[bcrypt]',
        'fastapi'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        package_name = package.split('[')[0]  # Handle passlib[bcrypt] format
        try:
            importlib.import_module(package_name)
            logger.info(f"✓ {package_name} is available")
        except ImportError:
            missing_packages.append(package)
            logger.warning(f"✗ {package_name} is missing")
    
    if missing_packages:
        logger.info(f"Installing missing packages: {', '.join(missing_packages)}")
        try:
            subprocess.check_call([
                sys.executable, '-m', 'pip', 'install', '--user'
            ] + missing_packages)
            logger.info("Dependencies installed successfully")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install dependencies: {e}")
            logger.error("Please install manually: pip install " + " ".join(missing_packages))
            return False
    else:
        logger.info("All dependencies are available")
    
    return True

def check_system_requirements():
    """Check system requirements for security testing"""
    checks_passed = 0
    total_checks = 4
    
    # Check Python version
    if sys.version_info >= (3, 7):
        logger.info("✓ Python version is compatible (>= 3.7)")
        checks_passed += 1
    else:
        logger.error("✗ Python version is too old. Requires Python 3.7 or newer")
    
    # Check for required files
    required_files = [
        'security_test_suite.py',
        'nsa_threat_simulation.py',
        'chaos_testing_agent.py',
        'security_test_orchestrator.py'
    ]
    
    for file_path in required_files:
        if Path(file_path).exists():
            logger.info(f"✓ {file_path} found")
            checks_passed += 1 / len(required_files)
        else:
            logger.error(f"✗ {file_path} not found")
    
    # Check network connectivity
    try:
        import socket
        socket.create_connection(("localhost", 8000), timeout=1)
        logger.info("✓ Target system is reachable (localhost:8000)")
        checks_passed += 1
    except (socket.error, OSError):
        logger.warning("⚠ Target system not reachable (localhost:8000) - tests may fail")
        logger.warning("  Ensure DSMIL backend is running or specify correct URL")
    
    # Check permissions
    try:
        test_file = Path("test_permissions.tmp")
        test_file.write_text("test")
        test_file.unlink()
        logger.info("✓ Write permissions available")
        checks_passed += 1
    except PermissionError:
        logger.error("✗ Insufficient write permissions for reports")
    
    success_rate = checks_passed / total_checks
    logger.info(f"System check: {success_rate:.1%} passed")
    
    return success_rate >= 0.75  # Need at least 75% to proceed

async def run_security_testing(target_url: str, test_phases: list, output_dir: str):
    """Run comprehensive security testing"""
    logger.info("DSMIL PHASE 3 SECURITY TESTING - STARTING")
    logger.info("Classification: RESTRICTED")
    logger.info("=" * 80)
    
    try:
        # Import security testing modules
        from security_test_orchestrator import SecurityTestOrchestrator
        
        # Initialize orchestrator
        orchestrator = SecurityTestOrchestrator(target_url)
        await orchestrator.initialize()
        
        # Execute comprehensive security assessment
        logger.info("Executing comprehensive security assessment...")
        assessment_results = await orchestrator.execute_comprehensive_security_assessment()
        
        # Generate reports
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Main assessment report
        main_report = Path(output_dir) / f"dsmil_security_assessment_{timestamp}.json"
        with open(main_report, 'w') as f:
            import json
            json.dump(assessment_results, f, indent=2, default=str)
        
        # Executive summary report
        exec_summary = generate_executive_summary(assessment_results)
        summary_report = Path(output_dir) / f"executive_summary_{timestamp}.md"
        with open(summary_report, 'w') as f:
            f.write(exec_summary)
        
        # Display results
        display_results(assessment_results)
        
        logger.info(f"Reports saved to:")
        logger.info(f"  Main report: {main_report}")
        logger.info(f"  Executive summary: {summary_report}")
        
        return assessment_results
        
    except Exception as e:
        logger.error(f"Security testing failed: {e}")
        raise
    finally:
        await orchestrator.cleanup()

def generate_executive_summary(assessment_results: dict) -> str:
    """Generate executive summary in markdown format"""
    security_posture = assessment_results.get("security_posture", {})
    risk_assessment = assessment_results.get("risk_assessment", {})
    consolidated_findings = assessment_results.get("consolidated_findings", {})
    recommendations = assessment_results.get("recommendations", [])
    
    summary = f"""# DSMIL Phase 3 Security Assessment - Executive Summary

**Classification:** RESTRICTED  
**Assessment Date:** {datetime.now().strftime('%Y-%m-%d')}  
**Target System:** DSMIL Phase 3 Multi-Client API  

## Overall Security Posture

- **Security Score:** {security_posture.get('security_score', 0)}/100
- **Security Grade:** {security_posture.get('security_grade', 'UNKNOWN')}  
- **Risk Level:** {risk_assessment.get('overall_risk_level', 'UNKNOWN')}

## Test Summary

- **Total Tests Executed:** {security_posture.get('total_tests_executed', 0)}
- **Test Pass Rate:** {security_posture.get('test_pass_rate', 0):.1f}%
- **Critical Vulnerabilities:** {security_posture.get('critical_vulnerabilities', 0)}

## Protection Status

- **Quarantine Protection:** {security_posture.get('quarantine_protection_status', 'UNKNOWN')}
- **Nation-State Resistance:** {security_posture.get('nation_state_resistance', 'UNKNOWN')}  
- **System Resilience:** {security_posture.get('system_resilience', 'UNKNOWN')}

## Phase Results

"""
    
    # Add phase results
    for phase_name, phase_data in assessment_results.get("phase_results", {}).items():
        if isinstance(phase_data, dict):
            phase_score = phase_data.get("security_score", 0)
            summary += f"- **{phase_name.title().replace('_', ' ')}:** {phase_score:.1f}%\n"
    
    summary += f"""
## Key Findings

"""
    
    # Add critical vulnerabilities
    if security_posture.get('critical_vulnerabilities', 0) > 0:
        summary += f"⚠️  **{security_posture['critical_vulnerabilities']} Critical Vulnerabilities Detected**\n\n"
    
    # Add quarantine status
    if security_posture.get('quarantine_protection_status') == 'COMPROMISED':
        summary += "🚨 **Quarantine Protection Compromised - Critical Security Breach**\n\n"
    
    # Add recommendations
    summary += "## Recommendations\n\n"
    for i, recommendation in enumerate(recommendations[:10], 1):  # Top 10 recommendations
        summary += f"{i}. {recommendation}\n"
    
    summary += f"""
## Risk Factors

"""
    
    for risk_factor in risk_assessment.get("risk_factors", []):
        summary += f"- {risk_factor}\n"
    
    summary += f"""
---
*This assessment was conducted using SECURITYAUDITOR, NSA threat simulation, and SECURITYCHAOSAGENT coordinated with BASTION defensive validation.*
"""
    
    return summary

def display_results(assessment_results: dict):
    """Display assessment results to console"""
    security_posture = assessment_results.get("security_posture", {})
    risk_assessment = assessment_results.get("risk_assessment", {})
    
    print("=" * 100)
    print("DSMIL PHASE 3 COMPREHENSIVE SECURITY ASSESSMENT - COMPLETE")
    print("=" * 100)
    print(f"Classification: RESTRICTED")
    print(f"Assessment Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Target System: DSMIL Phase 3 Multi-Client API")
    print("")
    print("EXECUTIVE SUMMARY:")
    print(f"  Overall Security Score: {security_posture.get('security_score', 0)}/100")
    print(f"  Security Grade: {security_posture.get('security_grade', 'UNKNOWN')}")
    print(f"  Risk Level: {risk_assessment.get('overall_risk_level', 'UNKNOWN')}")
    print(f"  Tests Executed: {security_posture.get('total_tests_executed', 0)}")
    print(f"  Test Pass Rate: {security_posture.get('test_pass_rate', 0):.1f}%")
    print(f"  Critical Vulnerabilities: {security_posture.get('critical_vulnerabilities', 0)}")
    print("")
    print("PROTECTION STATUS:")
    print(f"  Quarantine Protection: {security_posture.get('quarantine_protection_status', 'UNKNOWN')}")
    print(f"  Nation-State Resistance: {security_posture.get('nation_state_resistance', 'UNKNOWN')}")
    print(f"  System Resilience: {security_posture.get('system_resilience', 'UNKNOWN')}")
    print("")
    
    # Display phase results
    print("PHASE RESULTS:")
    for phase_name, phase_data in assessment_results.get("phase_results", {}).items():
        if isinstance(phase_data, dict):
            phase_score = phase_data.get("security_score", 0)
            phase_display = phase_name.replace('phase', 'Phase ').replace('_', ' ').title()
            print(f"  {phase_display}: {phase_score:.1f}%")
    print("")
    
    # Warnings
    if security_posture.get('critical_vulnerabilities', 0) > 0:
        print("⚠️  CRITICAL VULNERABILITIES DETECTED - IMMEDIATE ACTION REQUIRED")
    
    if security_posture.get('quarantine_protection_status') == 'COMPROMISED':
        print("🚨 QUARANTINE PROTECTION COMPROMISED - CRITICAL SECURITY BREACH")
    
    if security_posture.get('security_score', 0) < 70:
        print("⚠️  SECURITY SCORE BELOW ACCEPTABLE THRESHOLD")
    
    print("=" * 100)

async def run_individual_tests(target_url: str, test_type: str):
    """Run individual test components"""
    
    if test_type == "auditor":
        logger.info("Running SECURITYAUDITOR tests only...")
        from security_test_suite import DSMILSecurityTestSuite
        
        auditor = DSMILSecurityTestSuite(target_url, "v2")
        await auditor.initialize()
        
        try:
            # Run all auditor tests
            await auditor.test_authentication_security()
            await auditor.test_quarantine_protection()
            await auditor.test_api_penetration()
            await auditor.test_emergency_stop_security()
            await auditor.test_cross_client_security()
            
            # Generate report
            report = auditor.generate_security_report()
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_file = f"security_auditor_report_{timestamp}.json"
            
            with open(report_file, 'w') as f:
                import json
                json.dump(report, f, indent=2, default=str)
            
            logger.info(f"SECURITYAUDITOR report saved: {report_file}")
            
        finally:
            await auditor.cleanup()
    
    elif test_type == "nsa":
        logger.info("Running NSA threat simulation only...")
        from nsa_threat_simulation import NSAThreatSimulation
        
        nsa_sim = NSAThreatSimulation(target_url)
        await nsa_sim.initialize()
        
        try:
            # Run APT campaigns
            for actor in ["APT29", "Lazarus"]:
                await nsa_sim.execute_apt_campaign(actor, "TOP_SECRET")
            
            # Generate threat intelligence report
            intel_report = nsa_sim.generate_threat_intelligence_report()
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_file = f"nsa_threat_intel_{timestamp}.json"
            
            with open(report_file, 'w') as f:
                import json
                json.dump(intel_report, f, indent=2, default=str)
            
            logger.info(f"NSA threat intelligence report saved: {report_file}")
            
        finally:
            await nsa_sim.cleanup()
    
    elif test_type == "chaos":
        logger.info("Running SECURITYCHAOSAGENT tests only...")
        from chaos_testing_agent import SecurityChaosAgent
        
        chaos_agent = SecurityChaosAgent(target_url)
        await chaos_agent.initialize()
        
        try:
            # Run priority chaos experiments
            priority_experiments = [
                "auth_service_overload",
                "concurrent_quarantine_access",
                "emergency_stop_chaos"
            ]
            
            await chaos_agent.execute_chaos_campaign(priority_experiments)
            
            # Generate chaos report
            chaos_report = chaos_agent.generate_chaos_report()
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_file = f"chaos_testing_report_{timestamp}.json"
            
            with open(report_file, 'w') as f:
                import json
                json.dump(chaos_report, f, indent=2, default=str)
            
            logger.info(f"SECURITYCHAOSAGENT report saved: {report_file}")
            
        finally:
            await chaos_agent.cleanup()

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description="DSMIL Phase 3 Security Testing Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 run_security_tests.py                           # Full comprehensive testing
  python3 run_security_tests.py --url http://server:8000  # Test remote system
  python3 run_security_tests.py --test-type auditor       # Run only SECURITYAUDITOR
  python3 run_security_tests.py --test-type nsa           # Run only NSA simulation
  python3 run_security_tests.py --test-type chaos         # Run only chaos testing
  python3 run_security_tests.py --skip-deps               # Skip dependency check
  python3 run_security_tests.py --output-dir reports/     # Custom output directory

Classification: RESTRICTED
        """
    )
    
    parser.add_argument(
        '--url', 
        default='http://localhost:8000',
        help='Target DSMIL system URL (default: http://localhost:8000)'
    )
    
    parser.add_argument(
        '--test-type',
        choices=['full', 'auditor', 'nsa', 'chaos'],
        default='full',
        help='Type of security testing to run (default: full)'
    )
    
    parser.add_argument(
        '--output-dir',
        default='.',
        help='Output directory for reports (default: current directory)'
    )
    
    parser.add_argument(
        '--skip-deps',
        action='store_true',
        help='Skip dependency check and installation'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger.info("DSMIL Phase 3 Security Testing Suite")
    logger.info("Classification: RESTRICTED")
    logger.info("=" * 50)
    
    # Check dependencies
    if not args.skip_deps:
        logger.info("Checking dependencies...")
        if not check_and_install_dependencies():
            logger.error("Dependency check failed. Use --skip-deps to bypass.")
            sys.exit(1)
    
    # Check system requirements
    logger.info("Checking system requirements...")
    if not check_system_requirements():
        logger.warning("Some system checks failed. Tests may not run properly.")
        response = input("Continue anyway? (y/N): ")
        if response.lower() != 'y':
            logger.info("Exiting.")
            sys.exit(1)
    
    # Ensure output directory exists
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    logger.info(f"Configuration:")
    logger.info(f"  Target URL: {args.url}")
    logger.info(f"  Test Type: {args.test_type}")
    logger.info(f"  Output Directory: {output_dir.absolute()}")
    
    try:
        # Run security testing
        if args.test_type == 'full':
            asyncio.run(run_security_testing(args.url, [], str(output_dir)))
        else:
            asyncio.run(run_individual_tests(args.url, args.test_type))
            
        logger.info("Security testing completed successfully!")
        
    except KeyboardInterrupt:
        logger.warning("Security testing interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Security testing failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()