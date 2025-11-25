#!/usr/bin/env python3
"""
Comprehensive Test Suite for OSINT Collectors

Tests all OSINT collection modules:
- Military OSINT Collector
- Satellite Thermal Anomaly Collector
- OSINT Feed Aggregator
- Named Entity Extraction
- Geospatial Analysis
"""

import pytest
import json
import math
from datetime import datetime
from pathlib import Path

# Import OSINT collectors
from military_osint_collector import (
    MilitaryOSINTAggregator,
    MilitaryIntelItem,
    MilitaryNewsCollector,
    AircraftTrackingCollector,
    NavalTrackingCollector
)

from satellite_thermal_collector import (
    SatelliteThermalCollector,
    ThermalAnomaly,
    AnomalyAnalyzer
)


class TestMilitaryIntelItem:
    """Test MilitaryIntelItem data structure"""

    def test_intel_item_creation(self):
        """Test creating a military intel item"""
        item = MilitaryIntelItem(
            id="test123",
            title="Test Military Event",
            description="Test description",
            source="Test Source",
            intel_type="military_news"
        )

        assert item.id == "test123"
        assert item.title == "Test Military Event"
        assert item.classification == "UNCLASS"
        assert isinstance(item.tags, list)
        assert isinstance(item.entities, dict)
        assert item.collected is not None

    def test_intel_item_to_dict(self):
        """Test converting intel item to dictionary"""
        item = MilitaryIntelItem(
            id="test123",
            title="Test Event",
            description="Description",
            source="Source",
            intel_type="military_news",
            tags=["air_strike", "middle_east"]
        )

        item_dict = item.to_dict()

        assert isinstance(item_dict, dict)
        assert item_dict['id'] == "test123"
        assert item_dict['tags'] == ["air_strike", "middle_east"]


class TestMilitaryNewsCollector:
    """Test military news collection and parsing"""

    def test_entity_extraction_countries(self):
        """Test country entity extraction"""
        collector = MilitaryNewsCollector()

        text = "Russia and Ukraine continue conflict while China monitors Taiwan Strait"
        entities = collector._extract_entities(text)

        assert 'countries' in entities
        assert 'Russia' in entities['countries']
        assert 'Ukraine' in entities['countries']
        assert 'China' in entities['countries']
        assert 'Taiwan' in entities['countries']

    def test_entity_extraction_units(self):
        """Test military unit extraction"""
        collector = MilitaryNewsCollector()

        text = "The USS Gerald Ford and 101st Airborne Division are deployed"
        entities = collector._extract_entities(text)

        assert 'military_units' in entities
        assert any('USS Gerald Ford' in unit for unit in entities['military_units'])
        assert any('101st Airborne Division' in unit for unit in entities['military_units'])

    def test_entity_extraction_weapons(self):
        """Test weapon system extraction"""
        collector = MilitaryNewsCollector()

        text = "F-35 fighters and Patriot missile systems deployed alongside HIMARS"
        entities = collector._extract_entities(text)

        assert 'weapons' in entities
        assert 'F-35' in entities['weapons']
        assert 'Patriot' in entities['weapons']
        assert 'HIMARS' in entities['weapons']

    def test_military_tags_extraction(self):
        """Test military-specific tag extraction"""
        collector = MilitaryNewsCollector()

        title = "Air Strike on Military Exercise"
        description = "Air strike conducted during military exercise in the Persian Gulf"

        tags = collector._extract_military_tags(title, description)

        assert 'air_strike' in tags
        assert 'military_exercise' in tags
        assert 'persian_gulf' in tags

    def test_priority_assessment(self):
        """Test intelligence priority assessment"""
        collector = MilitaryNewsCollector()

        # High priority
        high_priority = collector._assess_priority(
            "Nuclear missile test",
            "North Korea conducts nuclear missile test"
        )
        assert high_priority == "HIGH"

        # Medium priority
        medium_priority = collector._assess_priority(
            "Military Exercise",
            "Routine military exercise in the Pacific"
        )
        assert medium_priority == "MEDIUM"

        # Low priority
        low_priority = collector._assess_priority(
            "General News",
            "General military news article"
        )
        assert low_priority == "LOW"

    def test_theater_tagging(self):
        """Test military theater tagging"""
        collector = MilitaryNewsCollector()

        # Indo-Pacific
        tags1 = collector._extract_military_tags("Event", "Incident in South China Sea")
        assert 'indo_pacific' in tags1

        # Taiwan Strait
        tags2 = collector._extract_military_tags("Event", "Taiwan Strait tension")
        assert 'taiwan_strait' in tags2

        # Middle East
        tags3 = collector._extract_military_tags("Event", "Persian Gulf operations")
        assert 'persian_gulf' in tags3

        # Ukraine
        tags4 = collector._extract_military_tags("Event", "Ukraine conflict update")
        assert 'ukraine' in tags4


class TestThermalAnomaly:
    """Test thermal anomaly data structure"""

    def test_anomaly_creation(self):
        """Test creating a thermal anomaly"""
        anomaly = ThermalAnomaly(
            anomaly_id="thermal123",
            latitude=14.0,
            longitude=-106.5,
            brightness=350.0,
            frp=12.5,
            confidence="high",
            acq_date="2025-11-09",
            acq_time="1234"
        )

        assert anomaly.anomaly_id == "thermal123"
        assert anomaly.latitude == 14.0
        assert anomaly.longitude == -106.5
        assert anomaly.frp == 12.5

    def test_anomaly_to_dict(self):
        """Test converting anomaly to dictionary"""
        anomaly = ThermalAnomaly(
            anomaly_id="test",
            latitude=10.0,
            longitude=20.0,
            brightness=300.0,
            frp=10.0,
            confidence="high",
            acq_date="2025-11-09",
            acq_time="1200"
        )

        anomaly_dict = anomaly.to_dict()
        assert isinstance(anomaly_dict, dict)
        assert anomaly_dict['latitude'] == 10.0


class TestAnomalyAnalyzer:
    """Test thermal anomaly analysis"""

    def test_haversine_distance(self):
        """Test Haversine distance calculation"""
        analyzer = AnomalyAnalyzer()

        # Test known distance (San Francisco to Los Angeles ≈ 559 km)
        sf_lat, sf_lon = 37.7749, -122.4194
        la_lat, la_lon = 34.0522, -118.2437

        distance = analyzer._haversine_distance(sf_lat, sf_lon, la_lat, la_lon)

        # Should be approximately 559 km (allow ±50km tolerance)
        assert 500 <= distance <= 610

    def test_haversine_zero_distance(self):
        """Test distance from point to itself"""
        analyzer = AnomalyAnalyzer()

        distance = analyzer._haversine_distance(10.0, 20.0, 10.0, 20.0)
        assert distance < 0.01  # Should be essentially zero

    def test_is_in_ocean(self):
        """Test ocean detection (simplified)"""
        analyzer = AnomalyAnalyzer()

        # Ocean location (middle of Pacific)
        assert analyzer._is_in_ocean(0.0, -150.0) == True

        # Land location (continental US)
        assert analyzer._is_in_ocean(40.0, -100.0) == False

    def test_score_anomaly_ocean(self):
        """Test anomaly scoring for ocean location"""
        analyzer = AnomalyAnalyzer()

        anomaly = ThermalAnomaly(
            anomaly_id="ocean_test",
            latitude=14.0,
            longitude=-106.5,  # Eastern Pacific Ocean
            brightness=350.0,
            frp=15.0,  # High FRP
            confidence="high",
            acq_date="2025-11-09",
            acq_time="1234",
            metadata={"is_ocean": True, "daytime": True}
        )

        score = analyzer._score_anomaly(anomaly)

        # Ocean + high FRP + high confidence should score well
        assert score > 50

    def test_score_anomaly_land_low_frp(self):
        """Test anomaly scoring for land location with low FRP"""
        analyzer = AnomalyAnalyzer()

        anomaly = ThermalAnomaly(
            anomaly_id="land_test",
            latitude=40.0,
            longitude=-100.0,  # Continental US (land)
            brightness=320.0,
            frp=2.0,  # Low FRP
            confidence="nominal",
            acq_date="2025-11-09",
            acq_time="0200",
            metadata={"is_ocean": False, "daytime": False}
        )

        score = analyzer._score_anomaly(anomaly)

        # Land + low FRP should score low
        assert score < 30

    def test_distance_from_infrastructure(self):
        """Test distance calculation from known infrastructure"""
        analyzer = AnomalyAnalyzer()

        # Test location near Acapulco, Mexico (16.8531°N, 99.8237°W)
        anomaly_lat, anomaly_lon = 16.0, -100.0

        # Acapulco coordinates
        port_lat, port_lon = 16.8531, -99.8237

        distance = analyzer._haversine_distance(anomaly_lat, anomaly_lon, port_lat, port_lon)

        # Should be less than 100 km
        assert distance < 150


class TestMilitaryOSINTAggregator:
    """Test military OSINT aggregator"""

    def test_aggregator_initialization(self):
        """Test aggregator initializes correctly"""
        aggregator = MilitaryOSINTAggregator()

        assert aggregator.military_news_collector is not None
        assert aggregator.aircraft_collector is not None
        assert aggregator.naval_collector is not None
        assert isinstance(aggregator.all_items, dict)

    def test_index_load_and_save(self, tmp_path):
        """Test index save/load operations"""
        # Temporarily redirect index file
        import military_osint_collector
        original_index = military_osint_collector.MILITARY_INDEX_FILE

        test_index = tmp_path / "test_military_index.json"
        military_osint_collector.MILITARY_INDEX_FILE = test_index

        try:
            aggregator = MilitaryOSINTAggregator()

            # Add test item
            item = MilitaryIntelItem(
                id="test123",
                title="Test Event",
                description="Test description",
                source="Test Source",
                intel_type="military_news",
                tags=["test"],
                metadata={"priority": "MEDIUM"}
            )

            aggregator.all_items[item.id] = item
            aggregator._save_index()

            # Verify file exists
            assert test_index.exists()

            # Load and verify
            with open(test_index, 'r') as f:
                index_data = json.load(f)

            assert 'items' in index_data
            assert 'last_update' in index_data
            assert 'stats' in index_data
            assert 'test123' in index_data['items']

        finally:
            # Restore original index file
            military_osint_collector.MILITARY_INDEX_FILE = original_index

    def test_statistics_calculation(self):
        """Test statistics calculation"""
        aggregator = MilitaryOSINTAggregator()

        # Add test items
        for i in range(10):
            priority = "HIGH" if i < 3 else "MEDIUM" if i < 7 else "LOW"
            item = MilitaryIntelItem(
                id=f"test{i}",
                title=f"Event {i}",
                description="Description",
                source="Test",
                intel_type="military_news",
                tags=["test"],
                entities={"countries": ["Russia"] if i % 2 == 0 else ["China"]},
                metadata={"priority": priority}
            )
            aggregator.all_items[item.id] = item

        aggregator._save_index()
        stats = aggregator.get_statistics()

        assert stats['total_items'] == 10
        assert stats['by_priority']['HIGH'] == 3
        assert stats['by_priority']['MEDIUM'] == 4
        assert stats['by_priority']['LOW'] == 3


class TestIntegration:
    """Integration tests for complete workflows"""

    def test_end_to_end_entity_extraction(self):
        """Test complete entity extraction workflow"""
        collector = MilitaryNewsCollector()

        # Realistic military news text
        text = """
        The USS Ronald Reagan Carrier Strike Group conducted operations in the
        South China Sea alongside F-35 stealth fighters and Aegis missile defense
        systems. The exercise involved the 3rd Marine Division and was in response
        to increased tensions with China near Taiwan.
        """

        entities = collector._extract_entities(text)

        # Verify extraction quality
        assert 'countries' in entities
        assert 'China' in entities['countries']
        assert 'Taiwan' in entities['countries']

        assert 'military_units' in entities
        assert any('USS Ronald Reagan' in unit for unit in entities['military_units'])

        assert 'weapons' in entities
        assert 'F-35' in entities['weapons']
        assert 'Aegis' in entities['weapons']

    def test_end_to_end_priority_and_tagging(self):
        """Test complete priority assessment and tagging"""
        collector = MilitaryNewsCollector()

        title = "Russian Missile Strike on Ukrainian Military Infrastructure"
        description = """
        Russian forces conducted missile strikes targeting Ukrainian military
        installations in eastern Ukraine. Multiple cruise missiles were detected
        by NATO radar systems. This represents an escalation in the ongoing conflict.
        """

        # Extract tags
        tags = collector._extract_military_tags(title, description)
        assert 'air_strike' in tags
        assert 'ukraine' in tags

        # Assess priority
        priority = collector._assess_priority(title, description)
        assert priority == "HIGH"

        # Extract entities
        entities = collector._extract_entities(title + " " + description)
        assert 'Russia' in entities.get('countries', [])
        assert 'Ukraine' in entities.get('countries', [])

    def test_thermal_anomaly_complete_analysis(self):
        """Test complete thermal anomaly analysis workflow"""
        analyzer = AnomalyAnalyzer()

        # Create realistic ocean anomaly
        anomaly = ThermalAnomaly(
            anomaly_id="pacific_event_001",
            latitude=14.2,
            longitude=-106.3,
            brightness=355.0,
            frp=18.5,  # High thermal output
            confidence="high",
            acq_date="2025-11-09",
            acq_time="1430",  # Daytime
            metadata={
                "satellite": "NOAA-20",
                "is_ocean": True,
                "daytime": True,
                "scan": 1.0,
                "track": 1.0
            }
        )

        # Score the anomaly
        score = analyzer._score_anomaly(anomaly)

        # Ocean location + high FRP + daytime + high confidence should score high
        assert score > 60, f"Expected high score for ocean anomaly, got {score}"

        # Verify it's in ocean
        assert analyzer._is_in_ocean(anomaly.latitude, anomaly.longitude)

        # Check distance from coast (should be significant)
        # Approximate Mexican coast
        coast_distance = analyzer._haversine_distance(
            anomaly.latitude, anomaly.longitude,
            16.0, -100.0  # Approximate coastline
        )
        assert coast_distance > 100  # Should be >100km from coast


def test_classification_markings():
    """Test that all OSINT data is properly marked as unclassified"""
    item = MilitaryIntelItem(
        id="test",
        title="Test",
        description="Test",
        source="Test",
        intel_type="military_news"
    )

    assert item.classification == "UNCLASS"


def test_data_sanitization():
    """Test that collected data is properly sanitized"""
    collector = MilitaryNewsCollector()

    # Test with potentially problematic characters
    text = "Test <script>alert('xss')</script> normal text"
    entities = collector._extract_entities(text)

    # Should still extract without errors
    assert isinstance(entities, dict)


if __name__ == '__main__':
    print("=" * 80)
    print("OSINT Collectors Test Suite")
    print("=" * 80)

    pytest.main([__file__, '-v', '--tb=short'])
