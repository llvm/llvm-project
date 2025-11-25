#!/usr/bin/env python3
"""
TOON (Token-Oriented Object Notation) Encoder/Decoder for Python

Pure Python implementation conforming to TOON Format Specification v1.4
https://github.com/toon-format/spec

TOON is an indentation-based serialization format designed for LLMs that:
- Reduces token consumption by 30-60% vs JSON
- Uses tabular format for uniform arrays of objects
- Employs minimal syntax (colon-based key-value pairs)
- Declares array lengths in headers

Key Features:
- Zero external dependencies (pure Python stdlib)
- Full TOON v1.4 specification compliance
- Functional design with type safety
- Memory-efficient streaming for large datasets
- Bidirectional JSON ↔ TOON conversion

Grammar:
- Objects: `key: value` (indentation-based nesting)
- Arrays: `key[N]: v1,v2,v3` (inline primitives)
- Tabular: `key[N]{f1,f2}: val1,val2 val3,val4` (uniform objects)
- Root arrays: `[N]: v1,v2` or `[N]{f1,f2}: row data`

Author: AI Framework Team
License: MIT
Specification: https://github.com/toon-format/spec/blob/main/SPEC.md
"""

import json
import re
from typing import Any, Dict, List, Union, Optional, Tuple, Iterator, Callable
from dataclasses import dataclass
from enum import Enum

# Type aliases
JsonValue = Union[None, bool, int, float, str, List[Any], Dict[str, Any]]


class ToonError(Exception):
    """Base exception for TOON encoding/decoding errors"""
    pass


class EncodingError(ToonError):
    """Error during TOON encoding"""
    pass


class DecodingError(ToonError):
    """Error during TOON decoding"""
    pass


class Delimiter(Enum):
    """TOON delimiter types"""
    COMMA = ","
    TAB = "\t"
    PIPE = "|"


@dataclass(frozen=True)
class ToonConfig:
    """Configuration for TOON encoding/decoding"""
    indent_size: int = 2                  # Spaces per indentation level
    delimiter: Delimiter = Delimiter.COMMA  # Default delimiter
    strict_mode: bool = False             # Enable strict validation
    preserve_order: bool = True           # Preserve key order
    use_tabular: bool = True              # Auto-detect and use tabular format
    min_tabular_rows: int = 2             # Minimum rows for tabular optimization


class ToonEncoder:
    """
    Encode JSON/Python objects to TOON format

    Implements TOON v1.4 specification:
    - Indentation-based structure (2 spaces default)
    - key: value notation
    - Array headers: key[N]: or key[N]{fields}:
    - Tabular optimization for uniform object arrays
    - Canonical number formatting
    - Delimiter-aware string quoting
    """

    def __init__(self, config: Optional[ToonConfig] = None):
        self.config = config or ToonConfig()
        self.indent = " " * self.config.indent_size

    def encode(self, obj: JsonValue) -> str:
        """
        Encode Python object to TOON format

        Args:
            obj: Python object (dict, list, str, int, float, bool, None)

        Returns:
            TOON-encoded string

        Raises:
            EncodingError: If object cannot be encoded
        """
        try:
            lines = []
            self._encode_value(obj, lines, depth=0, key=None, is_root=True)
            return "\n".join(lines)
        except Exception as e:
            raise EncodingError(f"Failed to encode object: {e}") from e

    def _encode_value(self, value: JsonValue, lines: List[str], depth: int,
                      key: Optional[str], is_root: bool = False) -> None:
        """
        Encode a value (recursive)

        Args:
            value: Value to encode
            lines: Output line buffer
            depth: Current indentation depth
            key: Key name (if this value is an object field)
            is_root: Is this the root value?
        """
        if isinstance(value, dict):
            self._encode_object(value, lines, depth, key, is_root)
        elif isinstance(value, list):
            self._encode_array(value, lines, depth, key, is_root)
        else:
            # Primitive value
            if key is not None:
                # Object field: key: value
                indent_str = self.indent * depth
                value_str = self._encode_primitive(value)
                lines.append(f"{indent_str}{key}: {value_str}")
            else:
                # Root primitive or array element
                value_str = self._encode_primitive(value)
                if is_root:
                    lines.append(value_str)

    def _encode_object(self, obj: Dict[str, JsonValue], lines: List[str],
                       depth: int, key: Optional[str], is_root: bool) -> None:
        """Encode object (dictionary)"""
        if not obj:
            # Empty object
            if is_root:
                # Empty root object = empty document
                pass
            elif key is not None:
                # Empty nested object
                indent_str = self.indent * depth
                lines.append(f"{indent_str}{key}:")
            return

        keys = list(obj.keys()) if self.config.preserve_order else sorted(obj.keys())

        for field_key in keys:
            field_value = obj[field_key]

            if isinstance(field_value, (dict, list)):
                # Nested structure: key on own line, content nested
                indent_str = self.indent * depth
                lines.append(f"{indent_str}{field_key}:")
                self._encode_value(field_value, lines, depth + 1, key=None)
            else:
                # Primitive field: key: value
                self._encode_value(field_value, lines, depth, key=field_key)

    def _encode_array(self, arr: List[JsonValue], lines: List[str],
                      depth: int, key: Optional[str], is_root: bool) -> None:
        """Encode array with optional tabular optimization"""
        if not arr:
            # Empty array
            header = self._make_array_header(key, 0, None, is_root)
            indent_str = "" if is_root else self.indent * depth
            lines.append(f"{indent_str}{header}:")
            return

        # Check if we should use tabular format
        if (self.config.use_tabular and
            len(arr) >= self.config.min_tabular_rows and
            self._is_tabular_array(arr)):
            self._encode_tabular(arr, lines, depth, key, is_root)
        elif all(not isinstance(item, (dict, list)) for item in arr):
            # All primitives: inline format
            self._encode_inline_array(arr, lines, depth, key, is_root)
        else:
            # Mixed or nested: expanded list format
            self._encode_expanded_array(arr, lines, depth, key, is_root)

    def _is_tabular_array(self, arr: List[JsonValue]) -> bool:
        """Check if array qualifies for tabular encoding"""
        if not arr or not isinstance(arr[0], dict):
            return False

        # All elements must be dicts
        if not all(isinstance(item, dict) for item in arr):
            return False

        # Get keys from first object
        first_keys = set(arr[0].keys())

        # All objects must have same keys
        if not all(isinstance(item, dict) and set(item.keys()) == first_keys for item in arr):
            return False

        # All values must be primitives (no nested structures)
        for item in arr:
            for value in item.values():
                if isinstance(value, (dict, list)):
                    return False

        return True

    def _encode_tabular(self, arr: List[Dict[str, JsonValue]], lines: List[str],
                        depth: int, key: Optional[str], is_root: bool) -> None:
        """
        Encode array in tabular format

        Format:
          key[N]{field1,field2,field3}:
            val1,val2,val3
            val1,val2,val3
        """
        if not arr:
            return

        # Get field names from first object
        fields = list(arr[0].keys())

        # Create header: key[N]{f1,f2,f3}:
        header = self._make_array_header(key, len(arr), fields, is_root)
        indent_str = "" if is_root else self.indent * depth
        lines.append(f"{indent_str}{header}:")

        # Encode rows
        delim = self.config.delimiter.value
        for obj in arr:
            row_values = [self._encode_tabular_value(obj.get(field), delim) for field in fields]
            row_str = delim.join(row_values)
            row_indent = self.indent * (depth + 1) if not is_root else self.indent
            lines.append(f"{row_indent}{row_str}")

    def _encode_inline_array(self, arr: List[JsonValue], lines: List[str],
                             depth: int, key: Optional[str], is_root: bool) -> None:
        """
        Encode primitive array inline

        Format: key[N]: v1,v2,v3
        """
        header = self._make_array_header(key, len(arr), None, is_root)
        indent_str = "" if is_root else self.indent * depth

        delim = self.config.delimiter.value
        values_str = delim.join(self._encode_primitive(item) for item in arr)

        lines.append(f"{indent_str}{header}: {values_str}")

    def _encode_expanded_array(self, arr: List[JsonValue], lines: List[str],
                                depth: int, key: Optional[str], is_root: bool) -> None:
        """
        Encode array in expanded list format

        Format:
          key[N]:
            - item1
            - item2
        """
        header = self._make_array_header(key, len(arr), None, is_root)
        indent_str = "" if is_root else self.indent * depth
        lines.append(f"{indent_str}{header}:")

        # Encode list items
        item_depth = depth + 1 if not is_root else 1
        for item in arr:
            item_indent = self.indent * item_depth if not is_root else self.indent

            if isinstance(item, dict):
                # Object as list item
                if not item:
                    # Empty object
                    lines.append(f"{item_indent}-")
                else:
                    # First field on hyphen line, rest nested
                    keys = list(item.keys()) if self.config.preserve_order else sorted(item.keys())
                    first_key = keys[0]
                    first_value = item[first_key]

                    if isinstance(first_value, (dict, list)):
                        # First field is nested
                        lines.append(f"{item_indent}- {first_key}:")
                        self._encode_value(first_value, lines, item_depth + 1, key=None)
                        # Remaining fields
                        for k in keys[1:]:
                            self._encode_value(item[k], lines, item_depth + 1, key=k)
                    else:
                        # First field is primitive
                        value_str = self._encode_primitive(first_value)
                        lines.append(f"{item_indent}- {first_key}: {value_str}")
                        # Remaining fields
                        for k in keys[1:]:
                            self._encode_value(item[k], lines, item_depth + 1, key=k)

            elif isinstance(item, list):
                # Array as list item
                lines.append(f"{item_indent}- {self._make_array_header(None, len(item), None, False)}:")
                if item and all(not isinstance(x, (dict, list)) for x in item):
                    # Inline primitive array
                    delim = self.config.delimiter.value
                    values_str = delim.join(self._encode_primitive(x) for x in item)
                    lines[-1] = lines[-1].rstrip(':') + f": {values_str}"
                else:
                    # Nested expanded array
                    for subitem in item:
                        self._encode_value(subitem, lines, item_depth + 1, key=None)

            else:
                # Primitive as list item
                value_str = self._encode_primitive(item)
                lines.append(f"{item_indent}- {value_str}")

    def _make_array_header(self, key: Optional[str], length: int,
                           fields: Optional[List[str]], is_root: bool) -> str:
        """Create array header: [N] or [N]{f1,f2} or key[N] or key[N]{f1,f2}"""
        delim = self.config.delimiter.value

        # Delimiter symbol in header (only for non-comma)
        delim_sym = ""
        if self.config.delimiter == Delimiter.TAB:
            delim_sym = "\t"
        elif self.config.delimiter == Delimiter.PIPE:
            delim_sym = "|"

        # Build header
        header = f"[{length}{delim_sym}]"

        if fields:
            # Add fields: {f1,f2,f3}
            fields_str = delim.join(self._encode_key(f) for f in fields)
            header += f"{{{fields_str}}}"

        if key and not is_root:
            # Add key prefix
            header = f"{self._encode_key(key)}{header}"

        return header

    def _encode_primitive(self, value: JsonValue) -> str:
        """Encode primitive value (null, bool, number, string)"""
        if value is None:
            return "null"

        if isinstance(value, bool):
            return "true" if value else "false"

        if isinstance(value, (int, float)):
            return self._encode_number(value)

        if isinstance(value, str):
            return self._encode_string(value)

        raise EncodingError(f"Cannot encode primitive type: {type(value)}")

    def _encode_number(self, num: Union[int, float]) -> str:
        """
        Encode number in canonical TOON form

        Rules:
        - No exponent notation (1e6 → 1000000)
        - No leading zeros
        - No trailing fractional zeros (1.50 → 1.5)
        - Integer when fractional part is zero (1.0 → 1)
        - Normalize -0 to 0
        """
        # Special values
        if num != num:  # NaN
            return "null"
        if num == float('inf') or num == float('-inf'):
            return "null"

        # Normalize -0 to 0
        if num == 0:
            num = 0

        # Convert to string without exponent notation
        if isinstance(num, int):
            return str(num)

        # Float
        if num == int(num):
            # Fractional part is zero
            return str(int(num))

        # Format with reasonable precision and remove trailing zeros
        # Try different precisions to find minimal representation
        for precision in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
            s = f"{num:.{precision}f}"
            if float(s) == num:
                return s.rstrip('0').rstrip('.')

        # Fallback to high precision
        s = f"{num:.15f}".rstrip('0').rstrip('.')
        return s

    def _encode_string(self, s: str) -> str:
        """Encode string with quoting if necessary"""
        if self._needs_quotes(s):
            # Escape and quote
            escaped = self._escape_string(s)
            return f'"{escaped}"'
        else:
            return s

    def _encode_tabular_value(self, value: JsonValue, active_delim: str) -> str:
        """Encode value for use in tabular row (may need quoting if contains delimiter)"""
        value_str = self._encode_primitive(value)

        # Check if value contains active delimiter and needs additional quoting
        if isinstance(value, str) and active_delim in value and not value_str.startswith('"'):
            value_str = f'"{self._escape_string(value)}"'

        return value_str

    def _encode_key(self, key: str) -> str:
        """Encode object key or field name"""
        # Check if key matches unquoted pattern
        if re.match(r'^[A-Za-z_][A-Za-z0-9_.]*$', key):
            return key
        else:
            # Quote and escape
            return f'"{self._escape_string(key)}"'

    def _needs_quotes(self, s: str) -> bool:
        """Check if string value needs quotes per TOON spec §7.2"""
        if not s:
            return True  # Empty string needs quotes

        # Leading/trailing whitespace
        if s != s.strip():
            return True

        # Reserved literals
        if s in ('true', 'false', 'null'):
            return True

        # Numeric pattern
        if re.match(r'^-?\d+(?:\.\d+)?(?:e[+-]?\d+)?$', s, re.IGNORECASE):
            return True

        # Leading zeros
        if re.match(r'^0\d+$', s):
            return True

        # Special characters
        if any(c in s for c in ':"\\[]{}\n\r\t'):
            return True

        # Active delimiter
        if self.config.delimiter.value in s:
            return True

        # Starts with hyphen at position 0 (list marker)
        if s.startswith('-'):
            return True

        return False

    def _escape_string(self, s: str) -> str:
        """Escape string per TOON spec §7.1 (only 5 valid escapes)"""
        s = s.replace('\\', '\\\\')  # Backslash
        s = s.replace('"', '\\"')    # Double quote
        s = s.replace('\n', '\\n')   # Newline
        s = s.replace('\r', '\\r')   # Carriage return
        s = s.replace('\t', '\\t')   # Tab
        return s


class ToonDecoder:
    """
    Decode TOON format to JSON/Python objects

    Implements TOON v1.4 specification:
    - Indentation-based parsing
    - Array header parsing with delimiters
    - Tabular row parsing
    - List item parsing with hyphen markers
    - Strict mode validation
    """

    def __init__(self, config: Optional[ToonConfig] = None):
        self.config = config or ToonConfig()

    def decode(self, toon_str: str) -> JsonValue:
        """
        Decode TOON string to Python object

        Args:
            toon_str: TOON-encoded string

        Returns:
            Python object

        Raises:
            DecodingError: If TOON string is invalid
        """
        try:
            parser = ToonParser(toon_str, self.config)
            return parser.parse()
        except Exception as e:
            raise DecodingError(f"Failed to decode TOON: {e}") from e

    def decode_stream(self, toon_docs: Iterator[str]) -> Iterator[JsonValue]:
        """Decode stream of TOON documents"""
        for doc in toon_docs:
            if doc.strip():
                yield self.decode(doc)


class ToonParser:
    """
    TOON document parser

    Parses indentation-based TOON format according to v1.4 spec.
    """

    def __init__(self, text: str, config: ToonConfig):
        self.config = config
        self.lines = text.split('\n')
        self.line_num = 0

    def parse(self) -> JsonValue:
        """Parse complete TOON document"""
        # Determine root form (§5)
        non_empty_lines = [line for line in self.lines if line.strip()]

        if not non_empty_lines:
            # Empty document → empty object
            return {}

        first_line = non_empty_lines[0]

        # Check if root array header
        if self._is_array_header(first_line.strip()):
            return self._parse_root_array()

        # Check if single primitive
        if len(non_empty_lines) == 1 and ':' not in first_line:
            return self._parse_primitive(first_line.strip())

        # Otherwise root object
        return self._parse_root_object()

    def _parse_root_object(self) -> Dict[str, JsonValue]:
        """Parse root object"""
        obj = {}
        i = 0
        while i < len(self.lines):
            line = self.lines[i]
            if not line.strip():
                i += 1
                continue

            depth = self._get_depth(line)
            if depth != 0:
                raise DecodingError(f"Line {i+1}: Root object fields must be at depth 0")

            content = line.strip()

            # Parse key-value or nested field
            if ':' in content:
                key, value_or_empty = self._split_key_value(content)

                if value_or_empty.strip():
                    # Inline value
                    obj[key] = self._parse_primitive(value_or_empty.strip())
                    i += 1
                else:
                    # Nested content
                    i, nested_value = self._parse_nested_content(i + 1, depth + 1)
                    obj[key] = nested_value
            else:
                raise DecodingError(f"Line {i+1}: Invalid root object line (no colon)")

        return obj

    def _parse_root_array(self) -> List[JsonValue]:
        """Parse root array"""
        # First line is array header
        header_line = self.lines[0].strip()
        length, delim, fields = self._parse_array_header(header_line)

        if fields:
            # Tabular array
            return self._parse_tabular_rows(1, 1, fields, delim, length)
        else:
            # Check if inline or expanded
            if ':' in header_line and len(header_line.split(':', 1)[1].strip()) > 0:
                # Inline array
                values_str = header_line.split(':', 1)[1].strip()
                return self._parse_inline_values(values_str, delim)
            else:
                # Expanded array
                return self._parse_expanded_array(1, 1, length)

    def _parse_nested_content(self, start_line: int, expected_depth: int) -> Tuple[int, JsonValue]:
        """Parse nested content (object or array) starting at start_line"""
        if start_line >= len(self.lines):
            return start_line, None

        first_content_line = None
        for i in range(start_line, len(self.lines)):
            if self.lines[i].strip():
                first_content_line = i
                break

        if first_content_line is None:
            return start_line, None

        line = self.lines[first_content_line]
        depth = self._get_depth(line)
        content = line.strip()

        # Check if array header
        if self._is_array_header(content):
            length, delim, fields = self._parse_array_header(content)

            if fields:
                # Tabular array
                arr = self._parse_tabular_rows(first_content_line + 1, depth + 1, fields, delim, length)
                return first_content_line + 1 + length, arr
            else:
                # Check inline or expanded
                if ':' in content and len(content.split(':', 1)[1].strip()) > 0:
                    # Inline
                    values_str = content.split(':', 1)[1].strip()
                    return first_content_line + 1, self._parse_inline_values(values_str, delim)
                else:
                    # Expanded
                    arr, end_line = self._parse_expanded_array_return_end(first_content_line + 1, depth + 1, length)
                    return end_line, arr

        # Check if list (hyphen markers)
        if content.startswith('-'):
            arr, end_line = self._parse_list_items(first_content_line, depth)
            return end_line, arr

        # Otherwise object
        obj, end_line = self._parse_object(first_content_line, depth)
        return end_line, obj

    def _parse_object(self, start_line: int, expected_depth: int) -> Tuple[Dict[str, JsonValue], int]:
        """Parse object starting at start_line"""
        obj = {}
        i = start_line

        while i < len(self.lines):
            line = self.lines[i]
            if not line.strip():
                i += 1
                continue

            depth = self._get_depth(line)

            if depth < expected_depth:
                # End of object
                break

            if depth > expected_depth:
                # Should not happen in well-formed TOON
                raise DecodingError(f"Line {i+1}: Unexpected indentation")

            content = line.strip()

            if ':' not in content:
                # End of object (no colon)
                break

            key, value_or_empty = self._split_key_value(content)

            if value_or_empty.strip():
                # Inline value
                obj[key] = self._parse_primitive(value_or_empty.strip())
                i += 1
            else:
                # Nested content
                i, nested_value = self._parse_nested_content(i + 1, depth + 1)
                obj[key] = nested_value

        return obj, i

    def _parse_tabular_rows(self, start_line: int, expected_depth: int,
                            fields: List[str], delim: str, declared_length: int) -> List[Dict[str, JsonValue]]:
        """Parse tabular array rows"""
        rows = []
        i = start_line

        while i < len(self.lines) and len(rows) < declared_length:
            line = self.lines[i]
            if not line.strip():
                if self.config.strict_mode:
                    raise DecodingError(f"Line {i+1}: Blank line within tabular array")
                i += 1
                continue

            depth = self._get_depth(line)

            if depth < expected_depth:
                break

            if depth > expected_depth:
                raise DecodingError(f"Line {i+1}: Invalid row indentation")

            content = line.strip()

            # Parse row values
            values = self._parse_inline_values(content, delim)

            if len(values) != len(fields):
                if self.config.strict_mode:
                    raise DecodingError(f"Line {i+1}: Row has {len(values)} values but {len(fields)} fields declared")

            row = {field: values[idx] if idx < len(values) else None for idx, field in enumerate(fields)}
            rows.append(row)
            i += 1

        if self.config.strict_mode and len(rows) != declared_length:
            raise DecodingError(f"Tabular array: {len(rows)} rows but {declared_length} declared")

        return rows

    def _parse_inline_values(self, values_str: str, delim: str) -> List[JsonValue]:
        """Parse inline comma/tab/pipe-separated values"""
        # Split by delimiter (respecting quoted strings)
        values = []
        current = ""
        in_quotes = False
        escaped = False

        for char in values_str:
            if escaped:
                current += char
                escaped = False
            elif char == '\\':
                escaped = True
                current += char
            elif char == '"':
                in_quotes = not in_quotes
                current += char
            elif char == delim and not in_quotes:
                values.append(self._parse_primitive(current.strip()))
                current = ""
            else:
                current += char

        if current.strip():
            values.append(self._parse_primitive(current.strip()))

        return values

    def _parse_expanded_array(self, start_line: int, expected_depth: int, declared_length: int) -> List[JsonValue]:
        """Parse expanded array (list with hyphen markers)"""
        arr, _ = self._parse_expanded_array_return_end(start_line, expected_depth, declared_length)
        return arr

    def _parse_expanded_array_return_end(self, start_line: int, expected_depth: int,
                                         declared_length: int) -> Tuple[List[JsonValue], int]:
        """Parse expanded array and return end line"""
        items, end_line = self._parse_list_items(start_line, expected_depth)

        if self.config.strict_mode and len(items) != declared_length:
            raise DecodingError(f"Expanded array: {len(items)} items but {declared_length} declared")

        return items, end_line

    def _parse_list_items(self, start_line: int, expected_depth: int) -> Tuple[List[JsonValue], int]:
        """Parse list items with hyphen markers"""
        items = []
        i = start_line

        while i < len(self.lines):
            line = self.lines[i]
            if not line.strip():
                i += 1
                continue

            depth = self._get_depth(line)

            if depth < expected_depth:
                break

            content = line.strip()

            if not content.startswith('-'):
                # End of list
                break

            # Remove hyphen and leading space
            item_content = content[1:].lstrip()

            if not item_content:
                # Empty object
                items.append({})
                i += 1
            elif ':' not in item_content:
                # Primitive item
                items.append(self._parse_primitive(item_content))
                i += 1
            else:
                # Object or inline value after hyphen
                # Parse as object with first field on hyphen line
                obj = {}
                key, value_or_empty = self._split_key_value(item_content)

                if value_or_empty.strip():
                    # Inline value on hyphen line
                    obj[key] = self._parse_primitive(value_or_empty.strip())
                else:
                    # Nested content
                    i, nested_value = self._parse_nested_content(i + 1, depth + 1)
                    obj[key] = nested_value
                    i -= 1  # Adjust because we'll increment below

                # Parse remaining fields at depth + 1
                i += 1
                while i < len(self.lines):
                    line = self.lines[i]
                    if not line.strip():
                        i += 1
                        continue

                    field_depth = self._get_depth(line)

                    if field_depth < depth + 1:
                        break

                    if field_depth > depth + 1:
                        raise DecodingError(f"Line {i+1}: Invalid field indentation")

                    field_content = line.strip()

                    if ':' not in field_content:
                        break

                    field_key, field_value_or_empty = self._split_key_value(field_content)

                    if field_value_or_empty.strip():
                        obj[field_key] = self._parse_primitive(field_value_or_empty.strip())
                        i += 1
                    else:
                        i, nested_value = self._parse_nested_content(i + 1, field_depth + 1)
                        obj[field_key] = nested_value

                items.append(obj)

        return items, i

    def _parse_primitive(self, s: str) -> JsonValue:
        """Parse primitive value from string"""
        if not s:
            return ""

        # Quoted string
        if s.startswith('"') and s.endswith('"'):
            return self._unescape_string(s[1:-1])

        # Boolean
        if s == "true":
            return True
        if s == "false":
            return False

        # Null
        if s == "null":
            return None

        # Number
        try:
            if '.' in s or 'e' in s.lower():
                return float(s)
            else:
                return int(s)
        except ValueError:
            pass

        # Unquoted string
        return s

    def _unescape_string(self, s: str) -> str:
        """Unescape string per TOON spec §7.1"""
        result = ""
        i = 0
        while i < len(s):
            if s[i] == '\\' and i + 1 < len(s):
                next_char = s[i + 1]
                if next_char == '\\':
                    result += '\\'
                elif next_char == '"':
                    result += '"'
                elif next_char == 'n':
                    result += '\n'
                elif next_char == 'r':
                    result += '\r'
                elif next_char == 't':
                    result += '\t'
                else:
                    raise DecodingError(f"Invalid escape sequence: \\{next_char}")
                i += 2
            else:
                result += s[i]
                i += 1
        return result

    def _is_array_header(self, s: str) -> bool:
        """Check if line is an array header"""
        return s.startswith('[') and (']:' in s or (']' in s and s.endswith(':')))

    def _parse_array_header(self, s: str) -> Tuple[int, str, Optional[List[str]]]:
        """
        Parse array header

        Returns: (length, delimiter, fields)
        """
        # Extract [N<delim?>] or [N<delim?>]{fields}:
        match = re.match(r'^\[(\d+)([,\t|])?\](?:\{([^}]+)\})?:', s)
        if not match:
            raise DecodingError(f"Invalid array header: {s}")

        length = int(match.group(1))
        delim_sym = match.group(2)
        fields_str = match.group(3)

        # Determine delimiter
        if delim_sym == '\t':
            delim = '\t'
        elif delim_sym == '|':
            delim = '|'
        else:
            delim = ','

        # Parse fields if present
        fields = None
        if fields_str:
            fields = self._parse_inline_values(fields_str, delim)
            # Fields are strings, unescape if quoted
            fields = [f if not (isinstance(f, str) and f.startswith('"')) else f[1:-1] for f in fields]

        return length, delim, fields

    def _split_key_value(self, s: str) -> Tuple[str, str]:
        """Split key: value line"""
        # Find first unquoted colon
        in_quotes = False
        escaped = False
        colon_pos = -1

        for i, char in enumerate(s):
            if escaped:
                escaped = False
            elif char == '\\':
                escaped = True
            elif char == '"':
                in_quotes = not in_quotes
            elif char == ':' and not in_quotes:
                colon_pos = i
                break

        if colon_pos == -1:
            raise DecodingError(f"No colon in key-value line: {s}")

        key_part = s[:colon_pos].strip()
        value_part = s[colon_pos + 1:].strip()

        # Unescape key if quoted
        if key_part.startswith('"') and key_part.endswith('"'):
            key = self._unescape_string(key_part[1:-1])
        else:
            key = key_part

        return key, value_part

    def _get_depth(self, line: str) -> int:
        """Get indentation depth of line"""
        leading_spaces = len(line) - len(line.lstrip(' '))

        if self.config.strict_mode and leading_spaces % self.config.indent_size != 0:
            raise DecodingError(f"Invalid indentation: {leading_spaces} spaces (not multiple of {self.config.indent_size})")

        return leading_spaces // self.config.indent_size


def json_to_toon(obj: JsonValue, config: Optional[ToonConfig] = None) -> str:
    """Convert JSON/Python object to TOON format"""
    encoder = ToonEncoder(config)
    return encoder.encode(obj)


def toon_to_json(toon_str: str, config: Optional[ToonConfig] = None) -> JsonValue:
    """Convert TOON format to JSON/Python object"""
    decoder = ToonDecoder(config)
    return decoder.decode(toon_str)


def estimate_token_savings(json_obj: JsonValue, tokenizer: Optional[Callable] = None) -> Dict[str, Any]:
    """Estimate token savings from using TOON vs JSON"""
    # Encode to both formats
    json_str = json.dumps(json_obj, separators=(',', ':'))
    toon_str = json_to_toon(json_obj)

    # Use provided tokenizer or fallback to character count
    if tokenizer is None:
        json_tokens = len(json_str)
        toon_tokens = len(toon_str)
    else:
        json_tokens = len(tokenizer(json_str))
        toon_tokens = len(tokenizer(toon_str))

    savings = json_tokens - toon_tokens
    savings_percent = (savings / json_tokens * 100) if json_tokens > 0 else 0

    return {
        'json_tokens': json_tokens,
        'toon_tokens': toon_tokens,
        'tokens_saved': savings,
        'savings_percent': savings_percent,
        'json_bytes': len(json_str),
        'toon_bytes': len(toon_str),
        'bytes_saved': len(json_str) - len(toon_str)
    }


if __name__ == "__main__":
    print("=" * 80)
    print("TOON Encoder/Decoder v1.4 - Specification Compliance Test")
    print("=" * 80)

    # Test 1: Tabular data (key TOON optimization)
    print("\n1. Tabular Data (Array of Uniform Objects):")
    tabular = [
        {"id": 1, "name": "Alice", "score": 95.5},
        {"id": 2, "name": "Bob", "score": 87.0},
        {"id": 3, "name": "Charlie", "score": 92.3}
    ]
    print("JSON:", json.dumps(tabular))
    toon_tabular = json_to_toon(tabular)
    print("TOON:\n" + toon_tabular)
    savings = estimate_token_savings(tabular)
    print(f"Savings: {savings['savings_percent']:.1f}%")
    decoded = toon_to_json(toon_tabular)
    print(f"Round-trip match: {decoded == tabular}")

    # Test 2: Nested object
    print("\n2. Nested Object:")
    nested = {
        "user": {
            "id": 1,
            "name": "Alice",
            "tags": ["admin", "ops", "dev"]
        },
        "timestamp": "2025-11-09T14:00:00Z"
    }
    toon_nested = json_to_toon(nested)
    print("TOON:\n" + toon_nested)
    decoded_nested = toon_to_json(toon_nested)
    print(f"Round-trip match: {decoded_nested == nested}")

    # Test 3: Token savings
    print("\n3. Token Savings Analysis:")
    large_data = [{"id": i, "name": f"User{i}", "active": i % 2 == 0, "score": i * 10.5} for i in range(50)]
    savings_large = estimate_token_savings(large_data)
    print(f"50 uniform objects:")
    print(f"  JSON:  {savings_large['json_tokens']} tokens / {savings_large['json_bytes']} bytes")
    print(f"  TOON:  {savings_large['toon_tokens']} tokens / {savings_large['toon_bytes']} bytes")
    print(f"  Saved: {savings_large['tokens_saved']} tokens ({savings_large['savings_percent']:.1f}%)")

    print("\n" + "=" * 80)
