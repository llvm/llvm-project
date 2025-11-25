#!/usr/bin/env python3
"""
Screenshot Intelligence CLI
Command-line interface for Screenshot Intelligence System

Commands:
- device: Manage devices
- ingest: Ingest screenshots and data
- search: Search intelligence database
- timeline: Query and generate timelines
- analyze: AI-powered analysis
- incident: Manage incidents
- stats: System statistics

Usage:
    screenshot-intel device register phone1 "GrapheneOS Phone 1" /path/to/screenshots
    screenshot-intel ingest scan phone1
    screenshot-intel search "VPN error" --limit 10
    screenshot-intel timeline 2025-11-10 2025-11-12 --format markdown
    screenshot-intel analyze 2025-11-10 2025-11-12 --detect-incidents
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Optional

# Add parent path
sys.path.insert(0, str(Path(__file__).parent))

# Core modules
try:
    from vector_rag_system import VectorRAGSystem
    from screenshot_intelligence import ScreenshotIntelligence
    from ai_analysis_layer import AIAnalysisLayer
    from telegram_integration import TelegramIntegration, TelegramConfig
    from signal_integration import SignalIntegration, SignalConfig
except ImportError as e:
    print(f"❌ Failed to import modules: {e}")
    print("Run setup script first: ./setup_screenshot_intel.sh")
    sys.exit(1)


class ScreenshotIntelCLI:
    """Screenshot Intelligence Command-Line Interface"""

    def __init__(self):
        """Initialize CLI"""
        self.intel = ScreenshotIntelligence()
        self.ai_analysis = AIAnalysisLayer(
            vector_rag=self.intel.rag,
            screenshot_intel=self.intel
        )

    def cmd_device_register(self, args):
        """Register a new device"""
        self.intel.register_device(
            device_id=args.device_id,
            device_name=args.name,
            device_type=args.type,
            screenshot_path=Path(args.path)
        )
        print(f"✓ Device registered: {args.name} ({args.device_id})")

    def cmd_device_list(self, args):
        """List registered devices"""
        if not self.intel.devices:
            print("No devices registered")
            return

        print(f"\n{'Device ID':<15} {'Name':<30} {'Type':<15} {'Path'}")
        print("=" * 90)
        for device_id, device in self.intel.devices.items():
            print(f"{device_id:<15} {device.device_name:<30} {device.device_type:<15} {device.screenshot_path}")
        print()

    def cmd_ingest_screenshot(self, args):
        """Ingest a single screenshot"""
        result = self.intel.ingest_screenshot(
            screenshot_path=Path(args.file),
            device_id=args.device
        )

        if result.get('status') == 'success':
            print(f"✓ Screenshot ingested: {result.get('id')}")
            if 'ocr_confidence' in result.get('metadata', {}):
                print(f"  OCR confidence: {result['metadata']['ocr_confidence']:.2%}")
        elif result.get('status') == 'already_indexed':
            print(f"⚠️  Screenshot already indexed: {result.get('id')}")
        else:
            print(f"❌ Error: {result.get('error', 'Unknown error')}")

    def cmd_ingest_scan(self, args):
        """Scan and ingest device screenshots"""
        result = self.intel.scan_device_screenshots(
            device_id=args.device_id,
            pattern=args.pattern
        )

        print(f"\n✓ Scan complete: {result['device_name']}")
        print(f"  Total files: {result['total']}")
        print(f"  Successfully ingested: {result['success']}")
        print(f"  Already indexed: {result['already_indexed']}")
        print(f"  Errors: {result['errors']}")

    def cmd_search(self, args):
        """Search intelligence database"""
        filters = {}
        if args.type:
            filters['type'] = args.type
        if args.source:
            filters['source'] = args.source
        if args.device:
            filters['device_id'] = args.device

        results = self.intel.rag.search(
            query=args.query,
            limit=args.limit,
            score_threshold=args.threshold,
            filters=filters if filters else None
        )

        if not results:
            print("No results found")
            return

        print(f"\n{'Score':<8} {'Type':<12} {'Timestamp':<20} {'Source':<20} {'Preview'}")
        print("=" * 120)

        for result in results:
            doc = result.document
            score_str = f"{result.score:.3f}"
            type_str = doc.doc_type[:11]
            time_str = doc.timestamp.strftime('%Y-%m-%d %H:%M:%S')
            source_str = (doc.metadata.get('device_name') or doc.metadata.get('source', 'Unknown'))[:19]
            preview = doc.text.replace('\n', ' ')[:60]

            print(f"{score_str:<8} {type_str:<12} {time_str:<20} {source_str:<20} {preview}")

        print(f"\nTotal results: {len(results)}\n")

    def cmd_timeline(self, args):
        """Query and generate timeline"""
        start_time = datetime.fromisoformat(args.start)
        end_time = datetime.fromisoformat(args.end)

        if args.format == 'json':
            events = self.intel.rag.timeline_query(start_time, end_time)
            timeline_data = [{
                'timestamp': e.timestamp.isoformat(),
                'type': e.doc_type,
                'source': e.metadata.get('source', 'unknown'),
                'text': e.text[:200]
            } for e in events]
            print(json.dumps(timeline_data, indent=2))

        else:  # markdown
            report = self.intel.generate_timeline_report(
                start_time, end_time, output_format='markdown'
            )
            print(report)

            if args.output:
                Path(args.output).write_text(report)
                print(f"\n✓ Report saved to: {args.output}")

    def cmd_analyze(self, args):
        """AI-powered timeline analysis"""
        start_time = datetime.fromisoformat(args.start)
        end_time = datetime.fromisoformat(args.end)

        print(f"Analyzing timeline: {args.start} to {args.end}")
        print("This may take a moment...\n")

        results = self.ai_analysis.analyze_timeline(
            start_time,
            end_time,
            auto_detect_incidents=args.detect_incidents
        )

        # Print analysis summary
        analysis = results['analysis']

        print("=" * 80)
        print(f"TIMELINE ANALYSIS REPORT")
        print("=" * 80)

        print(f"\n📊 STATISTICS")
        print(f"  Duration: {analysis['timeline']['duration_hours']:.1f} hours")
        print(f"  Total events: {analysis['statistics']['total_events']}")
        print(f"  Event types: {dict(analysis['statistics']['event_types'])}")

        print(f"\n🔗 EVENT LINKS")
        print(f"  Total links: {analysis['links']['total']}")
        print(f"  By type: {dict(analysis['links']['by_type'])}")
        print(f"  Avg confidence: {analysis['links']['avg_confidence']:.2%}")

        print(f"\n⚠️  ANOMALIES")
        print(f"  Total anomalies: {analysis['anomalies']['total']}")
        print(f"  High severity: {analysis['anomalies']['high_severity']}")
        print(f"  By type: {dict(analysis['anomalies']['by_type'])}")

        if results['anomalies']:
            print(f"\n  Top anomalies:")
            for i, anomaly in enumerate(sorted(results['anomalies'], key=lambda a: a.severity, reverse=True)[:5], 1):
                print(f"    {i}. [{anomaly.severity:.1%}] {anomaly.description}")

        print(f"\n📈 PATTERNS")
        print(f"  Total patterns: {analysis['patterns']['total']}")
        print(f"  By type: {dict(analysis['patterns']['by_type'])}")

        print(f"\n🚨 INCIDENTS")
        print(f"  Total incidents: {analysis['incidents']['total']}")
        print(f"  Auto-detected: {analysis['incidents']['auto_detected']}")

        if results['incidents']:
            print(f"\n  Detected incidents:")
            for i, incident in enumerate(results['incidents'], 1):
                print(f"    {i}. {incident.incident_name}")
                print(f"       Time: {incident.start_time.strftime('%Y-%m-%d %H:%M')} - {incident.end_time.strftime('%H:%M')}")
                print(f"       Events: {len(incident.events)}")

        print("\n" + "=" * 80)

        if args.output:
            Path(args.output).write_text(json.dumps({
                'analysis': analysis,
                'anomalies': [
                    {
                        'id': a.anomaly_id,
                        'type': a.anomaly_type,
                        'severity': a.severity,
                        'description': a.description,
                        'timestamp': a.timestamp.isoformat()
                    } for a in results['anomalies']
                ],
                'patterns': [
                    {
                        'id': p.pattern_id,
                        'type': p.pattern_type,
                        'frequency': p.frequency,
                        'description': p.description
                    } for p in results['patterns']
                ]
            }, indent=2))
            print(f"✓ Full analysis saved to: {args.output}")

    def cmd_incident_create(self, args):
        """Create an incident"""
        event_ids = args.event_ids.split(',')
        tags = args.tags.split(',') if args.tags else []

        incident = self.intel.create_incident(
            incident_name=args.name,
            event_ids=event_ids,
            tags=tags
        )

        print(f"✓ Incident created: {incident.incident_name}")
        print(f"  ID: {incident.incident_id}")
        print(f"  Events: {len(incident.events)}")

    def cmd_stats(self, args):
        """Display system statistics"""
        stats = self.intel.rag.get_stats()

        print("\n" + "=" * 80)
        print("SCREENSHOT INTELLIGENCE STATISTICS")
        print("=" * 80)

        print(f"\n📦 VECTOR DATABASE")
        print(f"  Collection: {stats['collection']}")
        print(f"  Total documents: {stats['total_documents']}")
        print(f"  Vector dimension: {stats['vector_dimension']}")
        print(f"  Embedding model: {stats['embedding_model']}")

        print(f"\n🖼️  OCR")
        print(f"  Engine: {stats['ocr_engine']}")

        print(f"\n📱 DEVICES")
        print(f"  Registered devices: {len(self.intel.devices)}")
        for device_id, device in self.intel.devices.items():
            print(f"    - {device.device_name} ({device.device_type})")

        print(f"\n🚨 INCIDENTS")
        print(f"  Total incidents: {len(self.intel.incidents)}")

        print("\n" + "=" * 80 + "\n")


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description='Screenshot Intelligence System CLI',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Device commands
    device_parser = subparsers.add_parser('device', help='Manage devices')
    device_subparsers = device_parser.add_subparsers(dest='subcommand')

    device_reg = device_subparsers.add_parser('register', help='Register a new device')
    device_reg.add_argument('device_id', help='Device identifier')
    device_reg.add_argument('name', help='Device name')
    device_reg.add_argument('type', choices=['grapheneos', 'laptop', 'pc'], help='Device type')
    device_reg.add_argument('path', help='Screenshot directory path')

    device_subparsers.add_parser('list', help='List registered devices')

    # Ingest commands
    ingest_parser = subparsers.add_parser('ingest', help='Ingest screenshots and data')
    ingest_subparsers = ingest_parser.add_subparsers(dest='subcommand')

    ingest_screenshot = ingest_subparsers.add_parser('screenshot', help='Ingest single screenshot')
    ingest_screenshot.add_argument('file', help='Screenshot file path')
    ingest_screenshot.add_argument('--device', help='Device ID')

    ingest_scan = ingest_subparsers.add_parser('scan', help='Scan device directory')
    ingest_scan.add_argument('device_id', help='Device ID')
    ingest_scan.add_argument('--pattern', default='*.png', help='File pattern (default: *.png)')

    # Search commands
    search_parser = subparsers.add_parser('search', help='Search intelligence database')
    search_parser.add_argument('query', help='Search query')
    search_parser.add_argument('--limit', type=int, default=10, help='Max results (default: 10)')
    search_parser.add_argument('--threshold', type=float, default=0.5, help='Score threshold (default: 0.5)')
    search_parser.add_argument('--type', help='Filter by type (image, chat_message, etc.)')
    search_parser.add_argument('--source', help='Filter by source (telegram, signal, etc.)')
    search_parser.add_argument('--device', help='Filter by device ID')

    # Timeline commands
    timeline_parser = subparsers.add_parser('timeline', help='Query and generate timelines')
    timeline_parser.add_argument('start', help='Start date (YYYY-MM-DD or YYYY-MM-DD HH:MM:SS)')
    timeline_parser.add_argument('end', help='End date (YYYY-MM-DD or YYYY-MM-DD HH:MM:SS)')
    timeline_parser.add_argument('--format', choices=['markdown', 'json'], default='markdown', help='Output format')
    timeline_parser.add_argument('--output', help='Output file')

    # Analyze commands
    analyze_parser = subparsers.add_parser('analyze', help='AI-powered timeline analysis')
    analyze_parser.add_argument('start', help='Start date (YYYY-MM-DD or YYYY-MM-DD HH:MM:SS)')
    analyze_parser.add_argument('end', help='End date (YYYY-MM-DD or YYYY-MM-DD HH:MM:SS)')
    analyze_parser.add_argument('--detect-incidents', action='store_true', help='Auto-detect incidents')
    analyze_parser.add_argument('--output', help='Output file (JSON)')

    # Incident commands
    incident_parser = subparsers.add_parser('incident', help='Manage incidents')
    incident_subparsers = incident_parser.add_subparsers(dest='subcommand')

    incident_create = incident_subparsers.add_parser('create', help='Create incident')
    incident_create.add_argument('name', help='Incident name')
    incident_create.add_argument('event_ids', help='Comma-separated event IDs')
    incident_create.add_argument('--tags', help='Comma-separated tags')

    # Stats command
    subparsers.add_parser('stats', help='Display system statistics')

    # Parse arguments
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    # Initialize CLI
    try:
        cli = ScreenshotIntelCLI()
    except Exception as e:
        print(f"❌ Failed to initialize: {e}")
        sys.exit(1)

    # Execute command
    try:
        if args.command == 'device':
            if args.subcommand == 'register':
                cli.cmd_device_register(args)
            elif args.subcommand == 'list':
                cli.cmd_device_list(args)
            else:
                device_parser.print_help()

        elif args.command == 'ingest':
            if args.subcommand == 'screenshot':
                cli.cmd_ingest_screenshot(args)
            elif args.subcommand == 'scan':
                cli.cmd_ingest_scan(args)
            else:
                ingest_parser.print_help()

        elif args.command == 'search':
            cli.cmd_search(args)

        elif args.command == 'timeline':
            cli.cmd_timeline(args)

        elif args.command == 'analyze':
            cli.cmd_analyze(args)

        elif args.command == 'incident':
            if args.subcommand == 'create':
                cli.cmd_incident_create(args)
            else:
                incident_parser.print_help()

        elif args.command == 'stats':
            cli.cmd_stats(args)

    except Exception as e:
        print(f"❌ Command failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
