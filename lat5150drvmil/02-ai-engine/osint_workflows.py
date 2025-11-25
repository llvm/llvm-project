#!/usr/bin/env python3
"""
OSINT Workflows - Pre-built investigation workflows

Provides pre-configured workflows for common OSINT investigations:
- Person investigation (social media, email, phone)
- Company/organization investigation (domain, employees, financials)
- Domain/infrastructure investigation (WHOIS, DNS, SSL certs)
- Email/phone investigation (validation, OSINT lookups)
- Comprehensive investigation (combines all sources)

Orchestrates multiple tools:
- DIRECTEYE Intelligence (40+ OSINT services)
- Entity Resolution Pipeline
- Threat Intelligence Automation
- Blockchain Investigation Tools

Part of Phase 4: Options B+C - Specialized Extensions
"""

import asyncio
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class WorkflowType(Enum):
    """Supported workflow types"""
    PERSON_INVESTIGATION = "person_investigation"
    COMPANY_INVESTIGATION = "company_investigation"
    DOMAIN_INVESTIGATION = "domain_investigation"
    EMAIL_PHONE_INVESTIGATION = "email_phone_investigation"
    COMPREHENSIVE_INVESTIGATION = "comprehensive_investigation"
    THREAT_ACTOR_INVESTIGATION = "threat_actor_investigation"
    CRYPTOCURRENCY_INVESTIGATION = "cryptocurrency_investigation"


class DataSource(Enum):
    """Data sources used in workflows"""
    DIRECTEYE = "directeye"
    ENTITY_RESOLUTION = "entity_resolution"
    THREAT_INTEL = "threat_intel"
    BLOCKCHAIN = "blockchain"
    WHOIS = "whois"
    DNS = "dns"
    SOCIAL_MEDIA = "social_media"
    SEARCH_ENGINE = "search_engine"
    BREACH_DATA = "breach_data"


@dataclass
class WorkflowStep:
    """Single workflow step"""
    step_number: int
    description: str
    data_source: DataSource
    query: str
    results: Optional[Dict] = None
    status: str = "pending"  # pending, running, completed, failed
    duration_seconds: float = 0.0


@dataclass
class WorkflowResult:
    """Complete workflow execution result"""
    workflow_id: str
    workflow_type: WorkflowType
    target: str
    steps: List[WorkflowStep]
    entities_found: List[Dict]
    findings: List[str]
    risk_assessment: Dict[str, Any]
    recommendations: List[str]
    data_sources_used: Set[DataSource]
    total_duration_seconds: float
    timestamp: datetime = field(default_factory=datetime.now)


class OSINTWorkflows:
    """
    Pre-built OSINT investigation workflows

    Orchestrates multiple intelligence sources to automate
    common investigation patterns.
    """

    def __init__(
        self,
        directeye_intel=None,
        entity_pipeline=None,
        threat_intel=None,
        blockchain_tools=None,
        event_driven_agent=None
    ):
        """
        Initialize OSINT workflows

        Args:
            directeye_intel: DIRECTEYE Intelligence Platform
            entity_pipeline: Entity Resolution Pipeline
            threat_intel: Threat Intelligence Automation
            blockchain_tools: Blockchain Investigation Tools
            event_driven_agent: Event-driven agent for audit logging
        """
        self.directeye_intel = directeye_intel
        self.entity_pipeline = entity_pipeline
        self.threat_intel = threat_intel
        self.blockchain_tools = blockchain_tools
        self.event_driven_agent = event_driven_agent

        # Statistics
        self.stats = {
            "workflows_executed": 0,
            "person_investigations": 0,
            "company_investigations": 0,
            "domain_investigations": 0,
            "email_phone_investigations": 0,
            "comprehensive_investigations": 0,
            "threat_actor_investigations": 0,
            "cryptocurrency_investigations": 0,
            "total_entities_found": 0,
        }

    async def investigate_person(
        self,
        name: Optional[str] = None,
        email: Optional[str] = None,
        phone: Optional[str] = None,
        username: Optional[str] = None
    ) -> WorkflowResult:
        """
        Person investigation workflow

        Steps:
        1. Entity extraction and resolution
        2. Social media lookup (username)
        3. Email validation and breach data
        4. Phone number lookup
        5. Search engine results
        6. Risk assessment

        Args:
            name: Person's name
            email: Email address
            phone: Phone number
            username: Social media username

        Returns:
            WorkflowResult with all findings
        """
        workflow_id = f"person_{datetime.now().timestamp()}"
        target = name or email or phone or username or "unknown"
        steps = []
        entities_found = []
        findings = []
        data_sources_used = set()

        start_time = datetime.now()

        # Step 1: Entity extraction
        if self.entity_pipeline:
            step = WorkflowStep(
                step_number=1,
                description="Extract and resolve entities",
                data_source=DataSource.ENTITY_RESOLUTION,
                query=target
            )
            steps.append(step)
            step.status = "running"

            # Would call entity_pipeline here
            step.results = {
                "entities": [],
                "note": "Entity extraction placeholder"
            }
            entities_found.extend(step.results.get("entities", []))
            data_sources_used.add(DataSource.ENTITY_RESOLUTION)
            step.status = "completed"

        # Step 2: Social media lookup
        if username and self.directeye_intel:
            step = WorkflowStep(
                step_number=2,
                description=f"Social media lookup for {username}",
                data_source=DataSource.SOCIAL_MEDIA,
                query=username
            )
            steps.append(step)
            step.status = "running"

            # Would call DIRECTEYE social media API
            step.results = {
                "platforms": ["twitter", "github", "linkedin"],
                "note": "Social media lookup placeholder"
            }
            findings.append(f"Found social media profiles for {username}")
            data_sources_used.add(DataSource.SOCIAL_MEDIA)
            step.status = "completed"

        # Step 3: Email investigation
        if email:
            step = WorkflowStep(
                step_number=3,
                description=f"Email validation and breach data for {email}",
                data_source=DataSource.BREACH_DATA,
                query=email
            )
            steps.append(step)
            step.status = "running"

            # Would call breach data API
            step.results = {
                "valid": True,
                "breaches": 0,
                "note": "Email investigation placeholder"
            }
            if step.results.get("breaches", 0) > 0:
                findings.append(f"Email found in {step.results['breaches']} data breaches")
            data_sources_used.add(DataSource.BREACH_DATA)
            step.status = "completed"

        # Step 4: Phone lookup
        if phone:
            step = WorkflowStep(
                step_number=4,
                description=f"Phone number lookup for {phone}",
                data_source=DataSource.DIRECTEYE,
                query=phone
            )
            steps.append(step)
            step.status = "running"

            # Would call phone lookup API
            step.results = {
                "carrier": "Unknown",
                "location": "Unknown",
                "note": "Phone lookup placeholder"
            }
            data_sources_used.add(DataSource.DIRECTEYE)
            step.status = "completed"

        # Risk assessment
        risk_assessment = {
            "overall_risk": "low",
            "factors": [],
            "data_breach_exposure": "low" if email else "unknown",
            "social_media_exposure": "medium" if username else "low"
        }

        # Recommendations
        recommendations = [
            "Monitor social media for updates",
            "Check for new data breaches regularly"
        ]

        total_duration = (datetime.now() - start_time).total_seconds()

        result = WorkflowResult(
            workflow_id=workflow_id,
            workflow_type=WorkflowType.PERSON_INVESTIGATION,
            target=target,
            steps=steps,
            entities_found=entities_found,
            findings=findings,
            risk_assessment=risk_assessment,
            recommendations=recommendations,
            data_sources_used=data_sources_used,
            total_duration_seconds=total_duration
        )

        # Update statistics
        self.stats["workflows_executed"] += 1
        self.stats["person_investigations"] += 1
        self.stats["total_entities_found"] += len(entities_found)

        # Log to event agent
        if self.event_driven_agent:
            self.event_driven_agent.log_event(
                event_type="tool_call",
                data={
                    "operation": "person_investigation",
                    "workflow_id": workflow_id,
                    "target": target,
                    "steps": len(steps),
                    "findings": len(findings)
                },
                metadata={"data_sources": [ds.value for ds in data_sources_used]}
            )

        return result

    async def investigate_company(
        self,
        company_name: Optional[str] = None,
        domain: Optional[str] = None
    ) -> WorkflowResult:
        """
        Company/organization investigation workflow

        Steps:
        1. Domain WHOIS lookup
        2. DNS enumeration
        3. SSL certificate analysis
        4. Email pattern detection
        5. Employee enumeration
        6. Threat intelligence check

        Args:
            company_name: Company name
            domain: Company domain

        Returns:
            WorkflowResult with all findings
        """
        workflow_id = f"company_{datetime.now().timestamp()}"
        target = company_name or domain or "unknown"
        steps = []
        entities_found = []
        findings = []
        data_sources_used = set()

        start_time = datetime.now()

        # Step 1: WHOIS lookup
        if domain:
            step = WorkflowStep(
                step_number=1,
                description=f"WHOIS lookup for {domain}",
                data_source=DataSource.WHOIS,
                query=domain
            )
            steps.append(step)
            step.status = "running"

            step.results = {
                "registrar": "Unknown",
                "registration_date": None,
                "note": "WHOIS lookup placeholder"
            }
            data_sources_used.add(DataSource.WHOIS)
            step.status = "completed"

        # Step 2: DNS enumeration
        if domain:
            step = WorkflowStep(
                step_number=2,
                description=f"DNS enumeration for {domain}",
                data_source=DataSource.DNS,
                query=domain
            )
            steps.append(step)
            step.status = "running"

            step.results = {
                "subdomains": [],
                "mx_records": [],
                "note": "DNS enumeration placeholder"
            }
            data_sources_used.add(DataSource.DNS)
            step.status = "completed"

        # Step 3: Threat intelligence check
        if self.threat_intel and domain:
            step = WorkflowStep(
                step_number=3,
                description=f"Threat intelligence check for {domain}",
                data_source=DataSource.THREAT_INTEL,
                query=domain
            )
            steps.append(step)
            step.status = "running"

            step.results = {
                "iocs": [],
                "threat_actors": [],
                "note": "Threat intel placeholder"
            }
            data_sources_used.add(DataSource.THREAT_INTEL)
            step.status = "completed"

        findings.append(f"Analyzed {len(steps)} data sources for {target}")

        risk_assessment = {
            "overall_risk": "low",
            "domain_age": "unknown",
            "threat_indicators": 0
        }

        recommendations = [
            "Monitor domain for changes",
            "Check SSL certificates regularly"
        ]

        total_duration = (datetime.now() - start_time).total_seconds()

        result = WorkflowResult(
            workflow_id=workflow_id,
            workflow_type=WorkflowType.COMPANY_INVESTIGATION,
            target=target,
            steps=steps,
            entities_found=entities_found,
            findings=findings,
            risk_assessment=risk_assessment,
            recommendations=recommendations,
            data_sources_used=data_sources_used,
            total_duration_seconds=total_duration
        )

        self.stats["workflows_executed"] += 1
        self.stats["company_investigations"] += 1

        if self.event_driven_agent:
            self.event_driven_agent.log_event(
                event_type="tool_call",
                data={
                    "operation": "company_investigation",
                    "workflow_id": workflow_id,
                    "target": target,
                    "steps": len(steps)
                },
                metadata={"data_sources": [ds.value for ds in data_sources_used]}
            )

        return result

    async def investigate_cryptocurrency(
        self,
        addresses: List[str]
    ) -> WorkflowResult:
        """
        Cryptocurrency investigation workflow

        Steps:
        1. Blockchain address analysis
        2. Fund flow tracing
        3. Address clustering
        4. Known address matching
        5. Risk assessment

        Args:
            addresses: List of cryptocurrency addresses

        Returns:
            WorkflowResult with all findings
        """
        workflow_id = f"crypto_{datetime.now().timestamp()}"
        target = ", ".join(addresses[:3]) + ("..." if len(addresses) > 3 else "")
        steps = []
        entities_found = []
        findings = []
        data_sources_used = set()

        start_time = datetime.now()

        # Step 1: Address analysis
        if self.blockchain_tools:
            step = WorkflowStep(
                step_number=1,
                description=f"Analyze {len(addresses)} cryptocurrency addresses",
                data_source=DataSource.BLOCKCHAIN,
                query=str(addresses)
            )
            steps.append(step)
            step.status = "running"

            # Generate investigation report
            report = self.blockchain_tools.generate_investigation_report(
                addresses=addresses,
                title="Cryptocurrency Investigation"
            )

            step.results = {
                "addresses_analyzed": len(addresses),
                "high_risk": report.risk_summary.get("high_risk", 0),
                "findings": report.findings
            }

            findings.extend(report.findings)
            data_sources_used.add(DataSource.BLOCKCHAIN)
            step.status = "completed"

            # Extract entities
            for addr in report.addresses_analyzed:
                entities_found.append({
                    "type": "cryptocurrency_address",
                    "value": addr.address,
                    "blockchain": addr.blockchain.value,
                    "risk_score": addr.risk_score
                })

        risk_assessment = {
            "overall_risk": step.results.get("risk_level", "unknown") if steps else "unknown",
            "high_risk_addresses": step.results.get("high_risk", 0) if steps else 0
        }

        recommendations = [
            "Monitor addresses for new transactions",
            "Check against updated threat feeds"
        ]

        total_duration = (datetime.now() - start_time).total_seconds()

        result = WorkflowResult(
            workflow_id=workflow_id,
            workflow_type=WorkflowType.CRYPTOCURRENCY_INVESTIGATION,
            target=target,
            steps=steps,
            entities_found=entities_found,
            findings=findings,
            risk_assessment=risk_assessment,
            recommendations=recommendations,
            data_sources_used=data_sources_used,
            total_duration_seconds=total_duration
        )

        self.stats["workflows_executed"] += 1
        self.stats["cryptocurrency_investigations"] += 1
        self.stats["total_entities_found"] += len(entities_found)

        if self.event_driven_agent:
            self.event_driven_agent.log_event(
                event_type="tool_call",
                data={
                    "operation": "cryptocurrency_investigation",
                    "workflow_id": workflow_id,
                    "addresses": len(addresses),
                    "findings": len(findings)
                },
                metadata={"data_sources": [ds.value for ds in data_sources_used]}
            )

        return result

    async def comprehensive_investigation(
        self,
        text: str,
        title: Optional[str] = None
    ) -> WorkflowResult:
        """
        Comprehensive investigation workflow

        Automatically detects entities in text and runs appropriate workflows:
        - Emails -> Email investigation
        - Phones -> Phone investigation
        - Domains -> Domain investigation
        - Crypto addresses -> Blockchain investigation
        - IOCs -> Threat intelligence

        Args:
            text: Text to analyze
            title: Optional investigation title

        Returns:
            WorkflowResult combining all investigations
        """
        workflow_id = f"comprehensive_{datetime.now().timestamp()}"
        target = title or "Comprehensive Investigation"
        steps = []
        entities_found = []
        findings = []
        data_sources_used = set()

        start_time = datetime.now()

        # Step 1: Entity extraction
        if self.entity_pipeline:
            step = WorkflowStep(
                step_number=1,
                description="Extract all entities from text",
                data_source=DataSource.ENTITY_RESOLUTION,
                query=text[:100] + "..."
            )
            steps.append(step)
            step.status = "running"

            # Would extract entities here
            step.results = {
                "entities": [],
                "note": "Entity extraction placeholder"
            }
            entities_found.extend(step.results.get("entities", []))
            data_sources_used.add(DataSource.ENTITY_RESOLUTION)
            step.status = "completed"

        # Step 2: Threat intelligence
        if self.threat_intel:
            step = WorkflowStep(
                step_number=2,
                description="Extract IOCs and analyze threats",
                data_source=DataSource.THREAT_INTEL,
                query=text[:100] + "..."
            )
            steps.append(step)
            step.status = "running"

            report = self.threat_intel.generate_report(
                title=title or "Threat Analysis",
                text=text
            )

            step.results = {
                "iocs": len(report.iocs),
                "threat_actors": report.threat_actors,
                "findings": report.summary
            }

            findings.append(f"Extracted {len(report.iocs)} IOCs")
            if report.threat_actors:
                findings.append(f"Attributed to: {', '.join(report.threat_actors)}")

            data_sources_used.add(DataSource.THREAT_INTEL)
            step.status = "completed"

        findings.append(f"Analyzed text using {len(data_sources_used)} data sources")

        risk_assessment = {
            "overall_risk": "medium",
            "entities_found": len(entities_found),
            "data_sources": len(data_sources_used)
        }

        recommendations = [
            "Review all extracted entities",
            "Cross-reference with known threat feeds"
        ]

        total_duration = (datetime.now() - start_time).total_seconds()

        result = WorkflowResult(
            workflow_id=workflow_id,
            workflow_type=WorkflowType.COMPREHENSIVE_INVESTIGATION,
            target=target,
            steps=steps,
            entities_found=entities_found,
            findings=findings,
            risk_assessment=risk_assessment,
            recommendations=recommendations,
            data_sources_used=data_sources_used,
            total_duration_seconds=total_duration
        )

        self.stats["workflows_executed"] += 1
        self.stats["comprehensive_investigations"] += 1
        self.stats["total_entities_found"] += len(entities_found)

        if self.event_driven_agent:
            self.event_driven_agent.log_event(
                event_type="tool_call",
                data={
                    "operation": "comprehensive_investigation",
                    "workflow_id": workflow_id,
                    "steps": len(steps),
                    "entities": len(entities_found),
                    "findings": len(findings)
                },
                metadata={"data_sources": [ds.value for ds in data_sources_used]}
            )

        return result

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics"""
        return self.stats.copy()


def demo():
    """Demo of OSINT workflows"""
    print("=== OSINT Workflows Demo ===\n")

    # Initialize
    workflows = OSINTWorkflows()

    print("1. Person Investigation Workflow")
    result = asyncio.run(workflows.investigate_person(
        name="John Doe",
        email="john.doe@example.com",
        username="johndoe"
    ))
    print(f"   Workflow ID: {result.workflow_id}")
    print(f"   Steps: {len(result.steps)}")
    print(f"   Findings: {len(result.findings)}")
    print(f"   Data sources: {[ds.value for ds in result.data_sources_used]}")
    print(f"   Duration: {result.total_duration_seconds:.2f}s")

    print("\n2. Company Investigation Workflow")
    result = asyncio.run(workflows.investigate_company(
        company_name="Example Corp",
        domain="example.com"
    ))
    print(f"   Workflow ID: {result.workflow_id}")
    print(f"   Steps: {len(result.steps)}")
    print(f"   Findings: {len(result.findings)}")

    print("\n3. Statistics:")
    stats = workflows.get_statistics()
    for key, value in stats.items():
        print(f"   {key}: {value}")


if __name__ == "__main__":
    demo()
