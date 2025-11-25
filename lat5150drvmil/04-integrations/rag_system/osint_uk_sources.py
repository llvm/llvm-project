#!/usr/bin/env python3
"""
UK-Specific OSINT Sources

Specialized OSINT collection for United Kingdom intelligence gathering.
Based on: https://github.com/paulpogoda/OSINT-Tools-UK

Categories:
- People Search: Electoral roll, birth/death records, genealogy
- Company Search: Companies House, financial registry
- Government Data: Census, statistics, open data
- Property: Land registry, rental data, mapping
- Vehicle: MOT records, number plates, aircraft
- Court Records: Judgments, appeals, tribunals
- Procurement: Government contracts and tenders
- Domain/Network: UK-specific WHOIS and registries

Usage:
    collector = UKOSINTCollector()

    # Search for person
    results = collector.search_person(name="John Smith", location="London")

    # Search company
    company = collector.search_company(name="Example Ltd")

    # Property lookup
    property_info = collector.search_property(postcode="SW1A 1AA")

    # Vehicle check
    vehicle = collector.check_vehicle(registration="ABC123")
"""

import logging
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


# ============================================================================
# PEOPLE SEARCH SOURCES
# ============================================================================

UK_PEOPLE_SEARCH = {
    "192_com": {
        "url": "https://www.192.com/",
        "description": "Electoral roll, director information, name/address lookup",
        "features": ["electoral_roll", "directors", "address_history"],
        "cost": "freemium",
        "note": "5 free searches/day"
    },

    "ReversePP": {
        "url": "https://reverseppp.com/",
        "description": "Reverse people search",
        "features": ["name_search", "address_search", "phone_search"],
        "cost": "freemium",
        "free_results": 5,
        "premium_cost": "£8/month"
    },

    "Public_Insights": {
        "url": "https://www.publicinsights.com/",
        "description": "People and business intelligence",
        "features": ["person_search", "business_links"],
        "cost": "commercial"
    },

    "Genes_Reunited": {
        "url": "https://www.genesreunited.co.uk/",
        "description": "Birth, marriage, death certificates and family tree",
        "features": ["bmd_records", "census", "genealogy"],
        "cost": "subscription"
    },

    "Amazon_UK_Wedding_Registry": {
        "url": "https://www.amazon.co.uk/wedding/home",
        "description": "Wedding plans and gift registries",
        "features": ["wedding_search", "couple_names", "date"],
        "cost": "free"
    },

    "FreeCen": {
        "url": "https://www.freecen.org.uk/",
        "description": "Historical census records (1841-1911)",
        "features": ["census_1841-1911", "genealogy"],
        "cost": "free"
    },

    "BMD_Registers": {
        "url": "https://www.bmdregisters.co.uk/",
        "description": "Non-conformist birth, marriage, death records",
        "features": ["nonconformist_records", "historical"],
        "cost": "paid"
    },

    "Find_My_Past": {
        "url": "https://www.findmypast.co.uk/",
        "description": "Family history and genealogy",
        "features": ["census", "military_records", "bmd"],
        "cost": "subscription"
    },

    "Ancestry_UK": {
        "url": "https://www.ancestry.co.uk/",
        "description": "Family tree and historical records",
        "features": ["census", "immigration", "military"],
        "cost": "subscription"
    }
}


# ============================================================================
# COMPANY & LEGAL ENTITIES
# ============================================================================

UK_COMPANY_SEARCH = {
    "Companies_House": {
        "url": "https://find-and-update.company-information.service.gov.uk/",
        "api": "https://developer-specs.company-information.service.gov.uk/",
        "description": "Official company registry for UK",
        "features": [
            "director_search_by_name",
            "director_search_by_location",
            "company_financials",
            "filing_history",
            "shareholders",
            "free_access"
        ],
        "cost": "free",
        "api_key_required": True
    },

    "Financial_Services_Register": {
        "url": "https://register.fca.org.uk/",
        "description": "FCA regulated financial firms and individuals",
        "features": ["financial_firms", "authorized_persons"],
        "cost": "free"
    },

    "Dun_and_Bradstreet": {
        "url": "https://www.dnb.co.uk/duns-number.html",
        "description": "D-U-N-S business identification numbers",
        "features": ["business_id", "credit_rating"],
        "cost": "commercial"
    },

    "MHRA_Medicine_Registry": {
        "url": "https://www.mhra.gov.uk/",
        "description": "Licensed medicine sellers registry",
        "features": ["pharmacy_licenses", "online_sellers"],
        "cost": "free"
    },

    "Charity_Commission": {
        "url": "https://register-of-charities.charitycommission.gov.uk/",
        "description": "Registered charities in England and Wales",
        "features": ["charity_search", "trustees", "income"],
        "cost": "free"
    },

    "Scottish_Charity_Register": {
        "url": "https://www.oscr.org.uk/",
        "description": "Scottish charity register",
        "features": ["scottish_charities"],
        "cost": "free"
    }
}


# ============================================================================
# GOVERNMENT DATABASES
# ============================================================================

UK_GOVERNMENT_DATA = {
    "Data_gov_uk": {
        "url": "https://www.data.gov.uk/",
        "description": "UK government open data portal",
        "features": ["datasets", "statistics", "maps"],
        "datasets": "50000+",
        "cost": "free"
    },

    "Office_for_National_Statistics": {
        "url": "https://www.ons.gov.uk/",
        "api": "https://developer.ons.gov.uk/",
        "description": "UK census and economic statistics",
        "features": ["census", "economy", "population"],
        "cost": "free"
    },

    "The_National_Archives": {
        "url": "https://www.nationalarchives.gov.uk/",
        "description": "Historical UK government records",
        "features": ["historical_docs", "cabinet_papers", "wills"],
        "cost": "free"
    },

    "BAILII": {
        "url": "https://www.bailii.org/",
        "description": "British and Irish Legal Information Institute",
        "features": ["legislation", "case_law", "judgments"],
        "cost": "free"
    },

    "Parliament_UK": {
        "url": "https://www.parliament.uk/",
        "description": "Parliamentary records and debates",
        "features": ["hansard", "bills", "mp_data"],
        "cost": "free"
    },

    "Electoral_Commission": {
        "url": "https://www.electoralcommission.org.uk/",
        "description": "Electoral data and party donations",
        "features": ["donations", "election_results"],
        "cost": "free"
    }
}


# ============================================================================
# PROPERTY & MAPS
# ============================================================================

UK_PROPERTY_SOURCES = {
    "Land_Registry": {
        "url": "https://www.gov.uk/government/organisations/land-registry",
        "description": "Property ownership records (England & Wales)",
        "features": ["ownership", "price_paid", "boundaries"],
        "cost": "£3_per_title"
    },

    "Cadastre_uk": {
        "url": "https://www.cadastre.uk/",
        "description": "England & Wales property mapping",
        "features": ["ownership_maps", "land_boundaries"],
        "cost": "freemium"
    },

    "Wales_Property_Register": {
        "url": "https://www.rentsmart.gov.wales/",
        "description": "Registered rental properties in Wales",
        "features": ["rental_properties", "landlord_registration"],
        "cost": "free"
    },

    "London_Rent_Maps": {
        "url": "https://www.london.gov.uk/",
        "description": "London rental pricing data",
        "features": ["rent_levels", "boroughs"],
        "cost": "free"
    },

    "Who_Owns_England": {
        "url": "https://whoownsengland.org/",
        "description": "Land ownership visualization and data",
        "features": ["land_ownership", "maps", "datasets"],
        "cost": "free"
    },

    "Rightmove": {
        "url": "https://www.rightmove.co.uk/",
        "description": "Property listings and sold prices",
        "features": ["current_listings", "sold_prices"],
        "cost": "free"
    },

    "Zoopla": {
        "url": "https://www.zoopla.co.uk/",
        "description": "Property valuations and sold prices",
        "features": ["valuations", "sold_prices", "rental"],
        "cost": "freemium"
    }
}


# ============================================================================
# VEHICLE RECORDS
# ============================================================================

UK_VEHICLE_SOURCES = {
    "Vehicle_Enquiry_Service": {
        "url": "https://www.gov.uk/get-vehicle-information-from-dvla",
        "description": "DVLA vehicle tax and MOT status",
        "features": ["tax_status", "mot_expiry", "make_model"],
        "cost": "free"
    },

    "Check_MOT_History": {
        "url": "https://www.check-mot.service.gov.uk/",
        "description": "Historical MOT test results",
        "features": ["mot_history", "mileage", "advisories"],
        "cost": "free"
    },

    "Partial_Number_Plate_Search": {
        "url": "https://www.gov.uk/vehicle-registration",
        "description": "Partial number plate lookup",
        "features": ["registration_lookup"],
        "cost": "varies"
    },

    "G_INFO_Aircraft_Register": {
        "url": "https://publicapps.caa.co.uk/modalapplication.aspx?appid=1",
        "description": "UK aircraft registration database",
        "features": ["aircraft_ownership", "registration"],
        "cost": "free"
    },

    "Ship_Register": {
        "url": "https://www.gov.uk/guidance/the-uks-ships-register",
        "description": "UK Ships Register",
        "features": ["vessel_registration", "ownership"],
        "cost": "official_use"
    }
}


# ============================================================================
# COURT RECORDS
# ============================================================================

UK_COURT_RECORDS = {
    "Courts_and_Tribunals_Judiciary": {
        "url": "https://www.judiciary.uk/",
        "description": "Court judgments and decisions",
        "features": ["judgments", "sentencing_remarks"],
        "cost": "free"
    },

    "Case_Tracker": {
        "url": "https://www.gov.uk/",
        "description": "Civil appeals and cases database",
        "features": ["civil_appeals", "case_status"],
        "cost": "free"
    },

    "Supreme_Court": {
        "url": "https://www.supremecourt.uk/",
        "description": "Supreme Court cases and judgments",
        "features": ["supreme_court_cases", "judgments"],
        "cost": "free"
    },

    "Scottish_Courts": {
        "url": "https://www.scotcourts.gov.uk/",
        "description": "Scottish court records",
        "features": ["scottish_judgments"],
        "cost": "free"
    },

    "Northern_Ireland_Courts": {
        "url": "https://www.judiciaryni.uk/",
        "description": "Northern Ireland court records",
        "features": ["ni_judgments"],
        "cost": "free"
    }
}


# ============================================================================
# PROCUREMENT
# ============================================================================

UK_PROCUREMENT_SOURCES = {
    "Contracts_Finder": {
        "url": "https://www.contractsfinder.service.gov.uk/",
        "description": "UK government contracts £12,000+",
        "features": ["tenders", "awards", "suppliers"],
        "threshold": "£12000+",
        "cost": "free"
    },

    "Find_a_Tender": {
        "url": "https://www.find-tender.service.gov.uk/",
        "description": "High-value procurement notices",
        "features": ["high_value_tenders", "framework_agreements"],
        "threshold": "£139688+",
        "cost": "free"
    },

    "Public_Contracts_Scotland": {
        "url": "https://www.publiccontractsscotland.gov.uk/",
        "description": "Scottish procurement portal",
        "features": ["scottish_tenders", "awards"],
        "cost": "free"
    },

    "Sell2Wales": {
        "url": "https://www.sell2wales.gov.wales/",
        "description": "Welsh procurement portal",
        "features": ["welsh_tenders", "supplier_registration"],
        "cost": "free"
    }
}


# ============================================================================
# DOMAIN & NETWORK
# ============================================================================

UK_DOMAIN_NETWORK = {
    "Nominet_UK": {
        "url": "https://www.nominet.uk/",
        "description": "UK domain registry (.uk, .co.uk)",
        "features": ["domain_registration", "whois"],
        "cost": "official"
    },

    "WHOIS_UK": {
        "url": "https://www.who.is/whois/",
        "description": "UK domain WHOIS lookup",
        "features": ["domain_ownership", "registration_dates"],
        "cost": "free"
    },

    "Jisc": {
        "url": "https://www.jisc.ac.uk/",
        "description": "UK academic/research network registry",
        "features": ["academic_domains", "research_networks"],
        "cost": "institutional"
    }
}


# ============================================================================
# Collector Class
# ============================================================================

@dataclass
class UKOSINTResult:
    """UK OSINT search result"""
    source: str
    category: str
    data: Dict
    timestamp: str
    query: str


class UKOSINTCollector:
    """
    UK-specific OSINT data collector

    Provides programmatic access to UK OSINT sources
    """

    def __init__(self):
        self.people_sources = UK_PEOPLE_SEARCH
        self.company_sources = UK_COMPANY_SEARCH
        self.government_sources = UK_GOVERNMENT_DATA
        self.property_sources = UK_PROPERTY_SOURCES
        self.vehicle_sources = UK_VEHICLE_SOURCES
        self.court_sources = UK_COURT_RECORDS
        self.procurement_sources = UK_PROCUREMENT_SOURCES
        self.domain_sources = UK_DOMAIN_NETWORK

        logger.info("UK OSINT Collector initialized")
        logger.info(f"  - {len(self.people_sources)} people search sources")
        logger.info(f"  - {len(self.company_sources)} company search sources")
        logger.info(f"  - {len(self.government_sources)} government data sources")
        logger.info(f"  - {len(self.property_sources)} property sources")
        logger.info(f"  - {len(self.vehicle_sources)} vehicle sources")
        logger.info(f"  - {len(self.court_sources)} court record sources")
        logger.info(f"  - {len(self.procurement_sources)} procurement sources")
        logger.info(f"  - {len(self.domain_sources)} domain/network sources")

    def get_all_sources(self) -> Dict:
        """Get all UK OSINT sources"""
        return {
            "people_search": self.people_sources,
            "company_search": self.company_sources,
            "government_data": self.government_sources,
            "property": self.property_sources,
            "vehicles": self.vehicle_sources,
            "courts": self.court_sources,
            "procurement": self.procurement_sources,
            "domain_network": self.domain_sources
        }

    def search_person(self, name: str, location: str = None) -> Dict:
        """
        Search for a person in UK databases

        Args:
            name: Person's name
            location: Optional location filter

        Returns:
            Dictionary of available search URLs and sources
        """
        logger.info(f"Person search: {name} (location: {location})")

        # Return URLs for manual searching (API integration would require keys)
        results = {}
        for source_name, source_data in self.people_sources.items():
            if source_data.get('cost') == 'free' or 'freemium' in source_data.get('cost', ''):
                results[source_name] = {
                    'url': source_data['url'],
                    'features': source_data['features'],
                    'note': f"Search for: {name}"
                }

        return results

    def search_company(self, name: str = None, company_number: str = None) -> Dict:
        """
        Search for a UK company

        Args:
            name: Company name
            company_number: Companies House number

        Returns:
            Dictionary of available search URLs
        """
        logger.info(f"Company search: {name or company_number}")

        results = {
            'Companies_House': {
                'url': 'https://find-and-update.company-information.service.gov.uk/',
                'search_name': name,
                'search_number': company_number,
                'api_available': True
            }
        }

        return results

    def search_property(self, postcode: str = None, address: str = None) -> Dict:
        """
        Search for property information

        Args:
            postcode: UK postcode
            address: Street address

        Returns:
            Dictionary of available property search sources
        """
        logger.info(f"Property search: {postcode or address}")

        return {
            'Land_Registry': self.property_sources['Land_Registry'],
            'Rightmove': self.property_sources['Rightmove'],
            'Zoopla': self.property_sources['Zoopla']
        }

    def check_vehicle(self, registration: str) -> Dict:
        """
        Check UK vehicle information

        Args:
            registration: Vehicle registration number

        Returns:
            Dictionary of vehicle check sources
        """
        logger.info(f"Vehicle check: {registration}")

        return {
            'DVLA': {
                'url': f"https://www.gov.uk/get-vehicle-information-from-dvla",
                'registration': registration
            },
            'MOT_History': {
                'url': f"https://www.check-mot.service.gov.uk/",
                'registration': registration
            }
        }

    def export_catalog(self, output_file: str = "uk_osint_catalog.json"):
        """Export UK OSINT source catalog"""
        import json
        from pathlib import Path

        catalog = self.get_all_sources()

        output_path = Path(output_file)
        with open(output_path, 'w') as f:
            json.dump(catalog, f, indent=2)

        logger.info(f"UK OSINT catalog exported to {output_path}")
        return output_path


def main():
    """Main function"""
    collector = UKOSINTCollector()

    # Export catalog
    catalog_file = collector.export_catalog()

    total_sources = sum([
        len(UK_PEOPLE_SEARCH),
        len(UK_COMPANY_SEARCH),
        len(UK_GOVERNMENT_DATA),
        len(UK_PROPERTY_SOURCES),
        len(UK_VEHICLE_SOURCES),
        len(UK_COURT_RECORDS),
        len(UK_PROCUREMENT_SOURCES),
        len(UK_DOMAIN_NETWORK)
    ])

    print("\n" + "=" * 80)
    print("UK-Specific OSINT Sources")
    print("=" * 80)
    print(f"\nTotal UK sources: {total_sources}")
    print(f"\nCatalog exported to: {catalog_file}")
    print("\nCategories:")
    print(f"  1. People Search - {len(UK_PEOPLE_SEARCH)} sources")
    print(f"  2. Company Search - {len(UK_COMPANY_SEARCH)} sources")
    print(f"  3. Government Data - {len(UK_GOVERNMENT_DATA)} sources")
    print(f"  4. Property - {len(UK_PROPERTY_SOURCES)} sources")
    print(f"  5. Vehicles - {len(UK_VEHICLE_SOURCES)} sources")
    print(f"  6. Court Records - {len(UK_COURT_RECORDS)} sources")
    print(f"  7. Procurement - {len(UK_PROCUREMENT_SOURCES)} sources")
    print(f"  8. Domain/Network - {len(UK_DOMAIN_NETWORK)} sources")
    print()


if __name__ == '__main__':
    main()
