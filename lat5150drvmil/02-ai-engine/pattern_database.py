#!/usr/bin/env python3
"""
Pattern Database for Local Claude Code
Stores and manages coding patterns, best practices, and learned knowledge
"""

import json
import time
import hashlib
import sqlite3
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from dataclasses import dataclass, field, asdict
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class PatternCategory(Enum):
    """Categories of coding patterns"""
    ALGORITHM = "algorithm"
    DATA_STRUCTURE = "data_structure"
    DESIGN_PATTERN = "design_pattern"
    BEST_PRACTICE = "best_practice"
    SECURITY = "security"
    PERFORMANCE = "performance"
    TESTING = "testing"
    ERROR_HANDLING = "error_handling"
    ARCHITECTURE = "architecture"
    IDIOM = "idiom"


class PatternQuality(Enum):
    """Quality levels for patterns"""
    EXCELLENT = "excellent"      # 5 stars
    GOOD = "good"               # 4 stars
    ACCEPTABLE = "acceptable"   # 3 stars
    POOR = "poor"              # 2 stars
    BAD = "bad"                # 1 star


@dataclass
class StoredPattern:
    """Pattern stored in database"""
    pattern_id: str
    name: str
    category: str
    quality: str
    description: str
    code_example: str
    language: str
    tags: List[str]
    usage_count: int = 0
    success_rate: float = 0.0
    source: str = "learned"  # 'learned', 'imported', 'curated'
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BestPractice:
    """Best practice guideline"""
    practice_id: str
    title: str
    description: str
    language: Optional[str] = None
    category: str = "general"
    do_example: Optional[str] = None
    dont_example: Optional[str] = None
    rationale: str = ""
    references: List[str] = field(default_factory=list)
    importance: int = 3  # 1-5, 5 being critical


class PatternDatabase:
    """
    Database for storing and retrieving coding patterns

    Features:
    - SQLite-based persistent storage
    - Pattern categorization and tagging
    - Quality scoring and usage tracking
    - Best practice management
    - Import/export capabilities
    - Integration with RAG system
    """

    def __init__(
        self,
        db_path: Optional[str] = None,
        rag_system=None,
        storage_system=None
    ):
        """
        Initialize pattern database

        Args:
            db_path: Path to SQLite database (default: .local_claude_code/patterns.db)
            rag_system: RAG system for semantic search
            storage_system: Storage orchestrator for advanced storage
        """
        if db_path:
            self.db_path = Path(db_path)
        else:
            self.db_path = Path.cwd() / ".local_claude_code" / "patterns.db"

        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self.rag = rag_system
        self.storage = storage_system

        # Initialize database
        self._init_database()

        logger.info(f"PatternDatabase initialized: {self.db_path}")

    def _init_database(self):
        """Initialize SQLite database schema"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        # Patterns table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS patterns (
                pattern_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                category TEXT NOT NULL,
                quality TEXT NOT NULL,
                description TEXT,
                code_example TEXT,
                language TEXT,
                tags TEXT,
                usage_count INTEGER DEFAULT 0,
                success_rate REAL DEFAULT 0.0,
                source TEXT DEFAULT 'learned',
                created_at REAL,
                updated_at REAL,
                metadata TEXT
            )
        ''')

        # Best practices table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS best_practices (
                practice_id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                description TEXT NOT NULL,
                language TEXT,
                category TEXT,
                do_example TEXT,
                dont_example TEXT,
                rationale TEXT,
                references TEXT,
                importance INTEGER DEFAULT 3,
                created_at REAL
            )
        ''')

        # Pattern usage history
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS pattern_usage (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pattern_id TEXT NOT NULL,
                timestamp REAL,
                success BOOLEAN,
                context TEXT,
                FOREIGN KEY (pattern_id) REFERENCES patterns(pattern_id)
            )
        ''')

        # Create indices
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_patterns_category ON patterns(category)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_patterns_language ON patterns(language)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_patterns_quality ON patterns(quality)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_best_practices_category ON best_practices(category)')

        conn.commit()
        conn.close()

        logger.info("Database schema initialized")

    def store_pattern(
        self,
        name: str,
        category: PatternCategory,
        code_example: str,
        description: str = "",
        language: str = "python",
        quality: PatternQuality = PatternQuality.ACCEPTABLE,
        tags: Optional[List[str]] = None,
        source: str = "learned",
        metadata: Optional[Dict] = None
    ) -> str:
        """
        Store a pattern in the database

        Args:
            name: Pattern name
            category: Pattern category
            code_example: Example code
            description: Pattern description
            language: Programming language
            quality: Quality level
            tags: Tags for categorization
            source: Pattern source (learned, imported, curated)
            metadata: Additional metadata

        Returns:
            Pattern ID
        """
        # Generate pattern ID
        pattern_id = hashlib.md5(
            f"{name}:{category.value}:{language}".encode()
        ).hexdigest()[:16]

        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        # Check if pattern exists
        cursor.execute('SELECT pattern_id FROM patterns WHERE pattern_id = ?', (pattern_id,))
        exists = cursor.fetchone()

        tags_json = json.dumps(tags or [])
        metadata_json = json.dumps(metadata or {})

        if exists:
            # Update existing pattern
            cursor.execute('''
                UPDATE patterns SET
                    description = ?,
                    code_example = ?,
                    quality = ?,
                    tags = ?,
                    updated_at = ?,
                    metadata = ?
                WHERE pattern_id = ?
            ''', (
                description,
                code_example,
                quality.value,
                tags_json,
                time.time(),
                metadata_json,
                pattern_id
            ))
        else:
            # Insert new pattern
            cursor.execute('''
                INSERT INTO patterns (
                    pattern_id, name, category, quality, description,
                    code_example, language, tags, source, created_at, updated_at, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                pattern_id,
                name,
                category.value,
                quality.value,
                description,
                code_example,
                language,
                tags_json,
                source,
                time.time(),
                time.time(),
                metadata_json
            ))

        conn.commit()
        conn.close()

        # Index in RAG if available
        if self.rag:
            self._index_pattern_in_rag(pattern_id, name, description, code_example, tags or [])

        logger.info(f"Pattern stored: {name} ({pattern_id})")

        return pattern_id

    def _index_pattern_in_rag(
        self,
        pattern_id: str,
        name: str,
        description: str,
        code_example: str,
        tags: List[str]
    ):
        """Index pattern in RAG system for semantic search"""
        try:
            # Create rich text for embedding
            text = f"{name}\n{description}\n\nCode:\n{code_example[:500]}\n\nTags: {', '.join(tags)}"

            self.rag.index_document(
                text=text,
                doc_id=f"pattern:{pattern_id}",
                metadata={
                    'type': 'pattern',
                    'pattern_id': pattern_id,
                    'name': name,
                    'tags': tags
                }
            )

        except Exception as e:
            logger.error(f"Error indexing pattern in RAG: {e}")

    def get_pattern(self, pattern_id: str) -> Optional[StoredPattern]:
        """Retrieve pattern by ID"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        cursor.execute('SELECT * FROM patterns WHERE pattern_id = ?', (pattern_id,))
        row = cursor.fetchone()
        conn.close()

        if not row:
            return None

        return self._row_to_pattern(row)

    def search_patterns(
        self,
        query: str,
        category: Optional[PatternCategory] = None,
        language: Optional[str] = None,
        min_quality: Optional[PatternQuality] = None,
        limit: int = 10
    ) -> List[StoredPattern]:
        """
        Search patterns by query

        Args:
            query: Search query
            category: Filter by category
            language: Filter by language
            min_quality: Minimum quality level
            limit: Maximum results

        Returns:
            List of matching patterns
        """
        # Try RAG first for semantic search
        if self.rag:
            try:
                results = self.rag.search(query, top_k=limit)

                pattern_ids = []
                for result in results:
                    if result.metadata.get('type') == 'pattern':
                        pattern_ids.append(result.metadata['pattern_id'])

                if pattern_ids:
                    patterns = [self.get_pattern(pid) for pid in pattern_ids]
                    return [p for p in patterns if p is not None]

            except Exception as e:
                logger.error(f"RAG search failed: {e}")

        # Fallback to SQL search
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        sql = 'SELECT * FROM patterns WHERE (name LIKE ? OR description LIKE ? OR tags LIKE ?)'
        params = [f'%{query}%', f'%{query}%', f'%{query}%']

        if category:
            sql += ' AND category = ?'
            params.append(category.value)

        if language:
            sql += ' AND language = ?'
            params.append(language)

        if min_quality:
            # Quality ordering: excellent > good > acceptable > poor > bad
            quality_order = {
                'excellent': 5, 'good': 4, 'acceptable': 3, 'poor': 2, 'bad': 1
            }
            min_score = quality_order.get(min_quality.value, 0)

            sql += ' AND ('
            quality_filters = []
            for q, score in quality_order.items():
                if score >= min_score:
                    quality_filters.append(f"quality = '{q}'")
            sql += ' OR '.join(quality_filters) + ')'

        sql += ' ORDER BY usage_count DESC, success_rate DESC LIMIT ?'
        params.append(limit)

        cursor.execute(sql, params)
        rows = cursor.fetchall()
        conn.close()

        return [self._row_to_pattern(row) for row in rows]

    def get_patterns_by_category(
        self,
        category: PatternCategory,
        limit: int = 20
    ) -> List[StoredPattern]:
        """Get all patterns in a category"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        cursor.execute('''
            SELECT * FROM patterns
            WHERE category = ?
            ORDER BY quality DESC, usage_count DESC
            LIMIT ?
        ''', (category.value, limit))

        rows = cursor.fetchall()
        conn.close()

        return [self._row_to_pattern(row) for row in rows]

    def get_best_practices(
        self,
        category: Optional[str] = None,
        language: Optional[str] = None,
        min_importance: int = 1
    ) -> List[BestPractice]:
        """Get best practices"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        sql = 'SELECT * FROM best_practices WHERE importance >= ?'
        params = [min_importance]

        if category:
            sql += ' AND category = ?'
            params.append(category)

        if language:
            sql += ' AND (language = ? OR language IS NULL)'
            params.append(language)

        sql += ' ORDER BY importance DESC, title'

        cursor.execute(sql, params)
        rows = cursor.fetchall()
        conn.close()

        practices = []
        for row in rows:
            references = json.loads(row[8]) if row[8] else []

            practices.append(BestPractice(
                practice_id=row[0],
                title=row[1],
                description=row[2],
                language=row[3],
                category=row[4],
                do_example=row[5],
                dont_example=row[6],
                rationale=row[7],
                references=references,
                importance=row[9]
            ))

        return practices

    def add_best_practice(
        self,
        title: str,
        description: str,
        category: str = "general",
        language: Optional[str] = None,
        do_example: Optional[str] = None,
        dont_example: Optional[str] = None,
        rationale: str = "",
        references: Optional[List[str]] = None,
        importance: int = 3
    ) -> str:
        """Add a best practice"""
        practice_id = hashlib.md5(
            f"{title}:{category}:{language or 'general'}".encode()
        ).hexdigest()[:16]

        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        references_json = json.dumps(references or [])

        cursor.execute('''
            INSERT OR REPLACE INTO best_practices (
                practice_id, title, description, language, category,
                do_example, dont_example, rationale, references, importance, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            practice_id,
            title,
            description,
            language,
            category,
            do_example,
            dont_example,
            rationale,
            references_json,
            importance,
            time.time()
        ))

        conn.commit()
        conn.close()

        logger.info(f"Best practice added: {title}")

        return practice_id

    def record_pattern_usage(self, pattern_id: str, success: bool, context: str = ""):
        """Record that a pattern was used"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        # Record usage
        cursor.execute('''
            INSERT INTO pattern_usage (pattern_id, timestamp, success, context)
            VALUES (?, ?, ?, ?)
        ''', (pattern_id, time.time(), success, context))

        # Update pattern statistics
        cursor.execute('SELECT usage_count, success_rate FROM patterns WHERE pattern_id = ?', (pattern_id,))
        row = cursor.fetchone()

        if row:
            usage_count, success_rate = row
            new_usage_count = usage_count + 1

            # Update success rate
            new_success_rate = (success_rate * usage_count + (1.0 if success else 0.0)) / new_usage_count

            cursor.execute('''
                UPDATE patterns SET
                    usage_count = ?,
                    success_rate = ?,
                    updated_at = ?
                WHERE pattern_id = ?
            ''', (new_usage_count, new_success_rate, time.time(), pattern_id))

        conn.commit()
        conn.close()

    def import_patterns_from_json(self, json_path: str) -> int:
        """Import patterns from JSON file"""
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)

            count = 0

            # Import patterns
            for pattern_data in data.get('patterns', []):
                self.store_pattern(
                    name=pattern_data['name'],
                    category=PatternCategory(pattern_data.get('category', 'idiom')),
                    code_example=pattern_data['code_example'],
                    description=pattern_data.get('description', ''),
                    language=pattern_data.get('language', 'python'),
                    quality=PatternQuality(pattern_data.get('quality', 'acceptable')),
                    tags=pattern_data.get('tags', []),
                    source=pattern_data.get('source', 'imported'),
                    metadata=pattern_data.get('metadata', {})
                )
                count += 1

            # Import best practices
            for practice_data in data.get('best_practices', []):
                self.add_best_practice(
                    title=practice_data['title'],
                    description=practice_data['description'],
                    category=practice_data.get('category', 'general'),
                    language=practice_data.get('language'),
                    do_example=practice_data.get('do_example'),
                    dont_example=practice_data.get('dont_example'),
                    rationale=practice_data.get('rationale', ''),
                    references=practice_data.get('references', []),
                    importance=practice_data.get('importance', 3)
                )
                count += 1

            logger.info(f"Imported {count} items from {json_path}")
            return count

        except Exception as e:
            logger.error(f"Error importing patterns: {e}")
            return 0

    def export_patterns_to_json(self, json_path: str) -> bool:
        """Export all patterns to JSON file"""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            # Get all patterns
            cursor.execute('SELECT * FROM patterns')
            pattern_rows = cursor.fetchall()

            patterns = []
            for row in pattern_rows:
                pattern = self._row_to_pattern(row)
                patterns.append(asdict(pattern))

            # Get all best practices
            cursor.execute('SELECT * FROM best_practices')
            practice_rows = cursor.fetchall()

            practices = []
            for row in practice_rows:
                practices.append({
                    'practice_id': row[0],
                    'title': row[1],
                    'description': row[2],
                    'language': row[3],
                    'category': row[4],
                    'do_example': row[5],
                    'dont_example': row[6],
                    'rationale': row[7],
                    'references': json.loads(row[8]) if row[8] else [],
                    'importance': row[9]
                })

            conn.close()

            # Write to file
            with open(json_path, 'w') as f:
                json.dump({
                    'patterns': patterns,
                    'best_practices': practices,
                    'exported_at': time.time()
                }, f, indent=2)

            logger.info(f"Exported {len(patterns)} patterns and {len(practices)} best practices to {json_path}")
            return True

        except Exception as e:
            logger.error(f"Error exporting patterns: {e}")
            return False

    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        cursor.execute('SELECT COUNT(*) FROM patterns')
        total_patterns = cursor.fetchone()[0]

        cursor.execute('SELECT COUNT(*) FROM best_practices')
        total_practices = cursor.fetchone()[0]

        cursor.execute('SELECT category, COUNT(*) FROM patterns GROUP BY category')
        patterns_by_category = dict(cursor.fetchall())

        cursor.execute('SELECT AVG(success_rate) FROM patterns WHERE usage_count > 0')
        avg_success_rate = cursor.fetchone()[0] or 0.0

        cursor.execute('SELECT SUM(usage_count) FROM patterns')
        total_usage = cursor.fetchone()[0] or 0

        conn.close()

        return {
            'total_patterns': total_patterns,
            'total_best_practices': total_practices,
            'patterns_by_category': patterns_by_category,
            'average_success_rate': avg_success_rate,
            'total_usage_count': total_usage
        }

    def _row_to_pattern(self, row) -> StoredPattern:
        """Convert database row to StoredPattern"""
        return StoredPattern(
            pattern_id=row[0],
            name=row[1],
            category=row[2],
            quality=row[3],
            description=row[4],
            code_example=row[5],
            language=row[6],
            tags=json.loads(row[7]) if row[7] else [],
            usage_count=row[8],
            success_rate=row[9],
            source=row[10],
            created_at=row[11],
            updated_at=row[12],
            metadata=json.loads(row[13]) if row[13] else {}
        )


def main():
    """Example usage"""
    print("=== Pattern Database Demo ===\n")

    db = PatternDatabase()

    print("1. Adding patterns...")

    # Add a pattern
    pattern_id = db.store_pattern(
        name="List Comprehension",
        category=PatternCategory.IDIOM,
        code_example="result = [x*2 for x in range(10) if x % 2 == 0]",
        description="Efficient way to create lists in Python",
        language="python",
        quality=PatternQuality.GOOD,
        tags=["python", "list", "comprehension", "efficient"],
        source="curated"
    )
    print(f"   Pattern added: {pattern_id}")

    # Add a best practice
    practice_id = db.add_best_practice(
        title="Use Type Hints",
        description="Always use type hints in Python 3.5+ for better code clarity and IDE support",
        category="typing",
        language="python",
        do_example="def greet(name: str) -> str:\n    return f'Hello, {name}'",
        dont_example="def greet(name):\n    return f'Hello, {name}'",
        rationale="Type hints improve code readability and enable static type checking",
        importance=4
    )
    print(f"   Best practice added: {practice_id}")

    print("\n2. Searching patterns...")
    results = db.search_patterns("list", language="python")
    print(f"   Found {len(results)} patterns")
    for p in results[:3]:
        print(f"   - {p.name} ({p.quality})")

    print("\n3. Getting best practices...")
    practices = db.get_best_practices(language="python", min_importance=3)
    print(f"   Found {len(practices)} practices")
    for p in practices[:3]:
        print(f"   - {p.title} (importance: {p.importance})")

    print("\n4. Database stats:")
    stats = db.get_database_stats()
    print(f"   Total patterns: {stats['total_patterns']}")
    print(f"   Total best practices: {stats['total_best_practices']}")
    print(f"   Categories: {stats['patterns_by_category']}")

    print("\n✓ Pattern Database ready!")


if __name__ == "__main__":
    main()
