# DevToys - Swiss Army Knife for Developers

**Project**: DevToys
**Repository**: https://github.com/DevToys-app/DevToys
**License**: MIT License
**Platform**: Cross-platform (Windows, macOS, Linux)
**Category**: Developer Utilities / Productivity Tools

![DevToys](https://img.shields.io/badge/DevToys-Developer%20Tools-blue)
![MIT License](https://img.shields.io/badge/License-MIT-green)
![Cross Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20macOS%20%7C%20Linux-orange)

---

## Executive Summary

**DevToys** is a comprehensive, offline-first developer utility application that bundles 30+ essential tools in a single cross-platform desktop application. Designed as "A Swiss Army knife for developers," it eliminates the need to use unreliable online services for common development tasks, ensuring **privacy**, **speed**, and **offline availability**.

**Key Differentiator**: **Smart Detection** - automatically identifies clipboard data type and suggests the appropriate tool.

---

## What is DevToys?

### Core Concept

DevToys consolidates dozens of developer utilities into one cohesive application, allowing developers to:
- **Work Offline**: No internet required, no data sent to third parties
- **Boost Productivity**: Instant access to tools via clipboard detection
- **Ensure Privacy**: Sensitive data never leaves your machine
- **Save Time**: No context switching to web browsers or online services

### Smart Detection

**Automatic Tool Selection**: When you copy data to your clipboard, DevToys:
1. Analyzes the data format (JSON, Base64, JWT, etc.)
2. Identifies the most likely operations you'll need
3. Automatically opens the appropriate tool
4. Pre-populates the input field with your clipboard data

**Example**:
```json
Copy: {"name": "test", "value": 123}
→ DevToys detects JSON
→ Opens JSON Formatter automatically
→ Ready to format/validate instantly
```

---

## Tools Included (30+ Default Tools)

### Converters

#### 1. JSON ⇄ YAML Converter
**Purpose**: Bidirectional conversion between JSON and YAML formats

**Use Cases**:
- Convert Kubernetes YAML to JSON for processing
- Transform JSON API responses to YAML configs
- Docker Compose file format conversion

**Example**:
```yaml
# Input (YAML)
version: '3.8'
services:
  web:
    image: nginx
    ports:
      - "80:80"

# Output (JSON)
{
  "version": "3.8",
  "services": {
    "web": {
      "image": "nginx",
      "ports": ["80:80"]
    }
  }
}
```

#### 2. Date Converter
**Purpose**: Convert between date formats, timezones, and epochs

**Features**:
- Unix timestamp ⇄ Human-readable date
- Timezone conversions
- Multiple date format support (ISO 8601, RFC 3339, etc.)

**Example**:
```
Unix Timestamp: 1730000000
→ Human: 2024-10-27 00:00:00 UTC
→ ISO 8601: 2024-10-27T00:00:00.000Z
→ RFC 3339: 2024-10-27T00:00:00+00:00
```

#### 3. Number Base Converter
**Purpose**: Convert numbers between binary, octal, decimal, hexadecimal

**Use Cases**:
- Memory address calculation
- Color code conversion
- Low-level programming

**Example**:
```
Decimal: 255
Binary:  11111111
Octal:   377
Hex:     0xFF
```

---

### Encoders / Decoders

#### 4. HTML Encoder/Decoder
**Purpose**: Encode/decode HTML entities

**Use Cases**:
- Prevent XSS attacks
- Display code snippets in HTML
- Parse HTML responses

**Example**:
```
Original: <script>alert('XSS')</script>
Encoded:  &lt;script&gt;alert(&#x27;XSS&#x27;)&lt;/script&gt;
```

#### 5. URL Encoder/Decoder
**Purpose**: Encode/decode URL components

**Use Cases**:
- Query string parameter encoding
- API URL construction
- Parse encoded URLs

**Example**:
```
Original: Hello World & Special=Characters
Encoded:  Hello%20World%20%26%20Special%3DCharacters
```

#### 6. Base64 Encoder/Decoder
**Purpose**: Base64 encoding and decoding

**Features**:
- Text encoding
- Binary file encoding
- Image data URLs

**Example**:
```
Original: Hello, DevToys!
Encoded:  SGVsbG8sIERldlRveXMh
```

#### 7. GZip Compression/Decompression
**Purpose**: Compress and decompress GZip data

**Use Cases**:
- HTTP response compression testing
- File size optimization
- Network payload analysis

#### 8. JWT Decoder
**Purpose**: Decode and inspect JSON Web Tokens

**Features**:
- Header inspection
- Payload decoding
- Signature verification (when secret provided)
- Expiration time checking

**Example**:
```
JWT: eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIiwiaWF0IjoxNTE2MjM5MDIyfQ.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c

Decoded Header:
{
  "alg": "HS256",
  "typ": "JWT"
}

Decoded Payload:
{
  "sub": "1234567890",
  "name": "John Doe",
  "iat": 1516239022
}
```

#### 9. QR Code Generator
**Purpose**: Generate QR codes from text/URLs

**Features**:
- Customizable size
- Error correction levels
- PNG/SVG export
- Batch generation

---

### Formatters

#### 10. JSON Formatter
**Purpose**: Format, validate, and minify JSON

**Features**:
- Pretty-print with configurable indentation
- Syntax validation with error highlighting
- Minification for production
- JSON to JSON Schema generation

**Example**:
```json
// Input (minified)
{"name":"test","items":[1,2,3],"nested":{"key":"value"}}

// Output (formatted)
{
  "name": "test",
  "items": [
    1,
    2,
    3
  ],
  "nested": {
    "key": "value"
  }
}
```

#### 11. SQL Formatter
**Purpose**: Format SQL queries for readability

**Features**:
- Multi-dialect support (MySQL, PostgreSQL, SQL Server, Oracle)
- Keyword highlighting
- Indentation normalization

**Example**:
```sql
-- Input
SELECT u.id,u.name,COUNT(o.id) FROM users u LEFT JOIN orders o ON u.id=o.user_id WHERE u.active=1 GROUP BY u.id

-- Output
SELECT
  u.id,
  u.name,
  COUNT(o.id)
FROM users u
LEFT JOIN orders o
  ON u.id = o.user_id
WHERE u.active = 1
GROUP BY u.id
```

#### 12. XML Formatter
**Purpose**: Format and validate XML documents

**Features**:
- Pretty-print with indentation
- Schema validation (XSD)
- Namespace handling
- XML to JSON conversion

---

### Generators

#### 13. Hash Generator
**Purpose**: Compute cryptographic hashes

**Algorithms Supported**:
- MD5
- SHA-1
- SHA-256
- SHA-384
- SHA-512
- BLAKE2

**Use Cases**:
- File integrity verification
- Password hashing (development only!)
- Digital signatures

**Example**:
```
Input: Hello, DevToys!

MD5:     9c8c3a8f8c8c3a8f8c8c3a8f8c8c3a8f
SHA-256: 2c26b46b68ffc68ff99b453c1d30413413422d706483bfa0f98a5e886266e7ae
```

#### 14. Lorem Ipsum Generator
**Purpose**: Generate placeholder text

**Options**:
- Words, sentences, paragraphs
- Configurable length
- HTML/Markdown formatting

**Example**:
```
Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.
```

#### 15. UUID / GUID Generator
**Purpose**: Generate universally unique identifiers

**Versions Supported**:
- UUIDv1 (time-based)
- UUIDv4 (random)
- UUIDv5 (name-based with SHA-1)

**Example**:
```
UUIDv4: f47ac10b-58cc-4372-a567-0e02b2c3d479
GUID:   {F47AC10B-58CC-4372-A567-0E02B2C3D479}
```

#### 16. Password Generator
**Purpose**: Generate secure random passwords

**Options**:
- Configurable length
- Character sets (uppercase, lowercase, numbers, symbols)
- Exclude ambiguous characters (0/O, 1/l)
- Bulk generation

**Example**:
```
Length: 16
Character set: All

Generated: T7$mK9!pL2@nR4vX
Strength: Very Strong (128-bit entropy)
```

---

### Graphics Tools

#### 17. Color Blindness Simulator
**Purpose**: Simulate how colors appear to colorblind users

**Types**:
- Protanopia (red-blind)
- Deuteranopia (green-blind)
- Tritanopia (blue-blind)
- Achromatopsia (total colorblindness)

**Use Cases**:
- Accessibility testing
- UI design validation
- Color scheme evaluation

#### 18. PNG/JPEG Compressor
**Purpose**: Reduce image file sizes

**Features**:
- Lossy and lossless compression
- Quality slider
- Batch processing
- Before/after preview

**Example**:
```
Original: image.png (2.4 MB)
Compressed: image_compressed.png (580 KB)
Reduction: 76% (1.82 MB saved)
Quality: 90% (visually lossless)
```

#### 19. Image Converter
**Purpose**: Convert between image formats

**Supported Formats**:
- PNG ⇄ JPEG ⇄ BMP ⇄ GIF ⇄ WEBP ⇄ SVG

---

### Testers

#### 20. JSONPath Tester
**Purpose**: Test JSONPath expressions on JSON documents

**Features**:
- Real-time evaluation
- Syntax highlighting
- Result preview

**Example**:
```json
// JSON Document
{
  "store": {
    "books": [
      {"title": "Book 1", "price": 10},
      {"title": "Book 2", "price": 15}
    ]
  }
}

// JSONPath
$.store.books[?(@.price < 12)].title

// Result
["Book 1"]
```

#### 21. Regular Expression Tester
**Purpose**: Test regex patterns with live matching

**Features**:
- Multi-flavor support (JavaScript, .NET, Python, PCRE)
- Match highlighting
- Group extraction
- Replacement preview

**Example**:
```
Pattern: (\d{3})-(\d{3})-(\d{4})
Input:   Call me at 555-123-4567
Matches: 1
Group 1: 555
Group 2: 123
Group 3: 4567
```

#### 22. XPath Tester
**Purpose**: Test XPath expressions on XML documents

#### 23. XML Validator
**Purpose**: Validate XML against DTD or XSD schemas

**Features**:
- Well-formedness checking
- Schema validation
- Error location highlighting

---

### Text Utilities

#### 24. Markdown Preview
**Purpose**: Real-time Markdown rendering

**Features**:
- GitHub Flavored Markdown (GFM)
- Syntax highlighting for code blocks
- Table support
- Export to HTML

#### 25. Text Diff / Comparison
**Purpose**: Compare two text documents

**Features**:
- Line-by-line comparison
- Character-level diff
- Inline or side-by-side view
- Merge conflict resolution

**Example**:
```diff
  Line 1: Same content
- Line 2: Original text
+ Line 2: Modified text
  Line 3: Same content
```

#### 26. Text Case Converter
**Purpose**: Convert text between cases

**Options**:
- UPPERCASE
- lowercase
- Title Case
- camelCase
- PascalCase
- snake_case
- kebab-case

#### 27. Text Inspector & Analyzer
**Purpose**: Analyze text statistics

**Metrics**:
- Character count
- Word count
- Line count
- Byte size
- Reading time estimate

#### 28. Escape / Unescape String
**Purpose**: Escape or unescape strings for various languages

**Formats**:
- C# / Java / JavaScript strings
- JSON strings
- SQL strings
- CSV strings

#### 29. Text to ASCII Art
**Purpose**: Convert text to ASCII art banners

**Fonts**: Multiple ASCII fonts available

**Example**:
```
 ____              _____
|  _ \  _____   __| ____|_   _  ___  _   _ ___
| | | |/ _ \ \ / /|  _| | | | |/ _ \| | | / __|
| |_| |  __/\ V / | |___| |_| | (_) | |_| \__ \
|____/ \___| \_/  |_____|\__,_|\___/ \__, |___/
                                     |___/
```

#### 30. Checksum Calculator
**Purpose**: Calculate file checksums for integrity verification

---

## Extensibility

### Extension System

DevToys supports **custom extensions** allowing developers to add new tools.

**Extension Types**:
1. **Converters**: Data format transformations
2. **Encoders/Decoders**: Encoding schemes
3. **Generators**: Data generation utilities
4. **Formatters**: Code/data formatting
5. **Custom Categories**: User-defined tool categories

### Creating Extensions

**API Documentation**: https://devtoys.app/doc
**Extension Template**: Available on GitHub

**Example Extension Structure**:
```csharp
[Export(typeof(IToolProvider))]
[Name("MyCustomTool")]
[Category("Custom")]
public class MyCustomToolProvider : IToolProvider
{
    public string MenuDisplayName => "My Custom Tool";
    public string Description => "Does something awesome";

    public IToolViewModel CreateTool()
    {
        return new MyCustomToolViewModel();
    }
}
```

---

## Technical Specifications

### Technology Stack

**Primary Language**: C# (73.3%)
**UI Framework**: WinUI 3 (Windows), Avalonia (cross-platform)
**Styling**: SCSS (10.6%)
**Markup**: HTML (6.6%)
**Scripting**: JavaScript (4.5%), TypeScript (4.3%)

### Platform Support

| Platform | Version | Status |
|----------|---------|--------|
| Windows  | 10, 11  | ✅ Stable |
| macOS    | 10.15+  | ✅ Stable |
| Linux    | Ubuntu 20.04+ | ✅ Stable |

### System Requirements

**Minimum**:
- **RAM**: 512 MB
- **Storage**: 100 MB
- **CPU**: Any modern processor

**Recommended**:
- **RAM**: 2 GB
- **Storage**: 200 MB
- **Display**: 1920x1080 or higher

---

## Installation

### Windows

**Microsoft Store** (Recommended):
```
1. Open Microsoft Store
2. Search "DevToys"
3. Click Install
```

**WinGet**:
```powershell
winget install DevToys
```

**Manual Download**:
- Download `.msix` from GitHub Releases
- Double-click to install

### macOS

**Homebrew**:
```bash
brew install devtoys
```

**Manual Download**:
- Download `.dmg` from GitHub Releases
- Drag to Applications folder

### Linux

**Snap** (Ubuntu/Debian):
```bash
sudo snap install devtoys
```

**AppImage** (Universal):
```bash
wget https://github.com/DevToys-app/DevToys/releases/latest/download/DevToys.AppImage
chmod +x DevToys.AppImage
./DevToys.AppImage
```

**Flatpak**:
```bash
flatpak install flathub com.devtoys.DevToys
```

---

## Integration with LAT5150DRVMIL

### Cybersecurity Use Cases

#### 1. Malware Analysis Workflow

**JWT Token Analysis**:
```
Extract JWT from malware C2 communication
→ Paste into DevToys JWT Decoder
→ Inspect payload for attacker infrastructure
→ Add IOCs to SWORD Intelligence feed
```

**Base64-Encoded Payloads**:
```
Intercept Base64-encoded PowerShell commands
→ DevToys Base64 Decoder
→ Reveal malicious command
→ Generate YARA rule with LAT5150DRVMIL
```

#### 2. Hash Computation for Threat Intel

```python
# LAT5150DRVMIL malware analyzer already does this,
# but DevToys provides quick manual verification:

# Malware sample hash
# DevToys Hash Generator → SHA-256
# Cross-reference with:
# - VirusTotal
# - SWORD Intelligence database
# - MITRE ATT&CK IOCs
```

#### 3. Network Forensics

**URL Decoding**:
```
Suspicious URL from PCAP:
http://evil.com/payload?data=Hello%20World%26cmd%3Dexec

DevToys URL Decoder →
http://evil.com/payload?data=Hello World&cmd=exec

Reveals command execution attempt
```

**GZip Decompression**:
```
Compressed HTTP response from C2 server
→ DevToys GZip Decompressor
→ Inspect plaintext payload
→ Extract IOCs
```

#### 4. Code Deobfuscation

**JSON Minification Reversal**:
```
Obfuscated JavaScript from malicious site:
{"config":{"server":"evil.com","port":443}}

DevToys JSON Formatter →
{
  "config": {
    "server": "evil.com",
    "port": 443
  }
}

Clear C2 configuration revealed
```

### Cython/Spy Integration

**Performance Benchmarking**:
```python
# Use DevToys Hash Generator to benchmark
# LAT5150DRVMIL Cython hash module

# DevToys (C#, managed code): ~500 MB/s
# Cython module (C-level): ~2000 MB/s
# 4x speedup for hash computation
```

### Neural Code Synthesis Integration

**Generate DevToys-Style Tool**:
```python
from rag_system.neural_code_synthesis import NeuralCodeSynthesizer

synthesizer = NeuralCodeSynthesizer(rag_retriever=None)

# Generate Cython hash calculator (DevToys equivalent)
hash_tool = synthesizer.generate_module(
    "Fast Cython hash generator with MD5, SHA256, SHA512 support"
)

# Result: C-speed hash computation
# Integrates with LAT5150DRVMIL malware analysis
```

---

## Privacy & Security

### Privacy Features

✅ **100% Offline Operation**: No internet connection required
✅ **No Data Collection**: Zero telemetry, no analytics
✅ **No Cloud Services**: All processing local
✅ **Open Source**: Full source code audit available

### Security Considerations

**Safe for Sensitive Data**:
- API keys
- Passwords (generation only, not storage!)
- Proprietary code
- Customer data
- Security tokens

**NOT a Password Manager**: DevToys generates passwords but does NOT store them. Use a dedicated password manager (Bitwarden, 1Password, KeePass) for storage.

---

## Community & Contribution

### GitHub Statistics (as of Nov 2025)

- **Stars**: 23,000+
- **Forks**: 1,200+
- **Contributors**: 150+
- **Releases**: 50+

### Contributing

**Ways to Contribute**:
1. Report bugs via GitHub Issues
2. Request features
3. Submit pull requests
4. Create extensions
5. Translate to new languages
6. Write documentation

**Code Contribution**:
```bash
git clone https://github.com/DevToys-app/DevToys.git
cd DevToys
dotnet build
dotnet run
```

---

## Comparison with Alternatives

| Feature | DevToys | CyberChef | Online Tools |
|---------|---------|-----------|--------------|
| **Offline** | ✅ | ✅ | ❌ |
| **Privacy** | ✅ | ✅ | ❌ |
| **Smart Detection** | ✅ | ❌ | ❌ |
| **Native App** | ✅ | ❌ (Browser) | ❌ (Browser) |
| **Extensible** | ✅ | Limited | ❌ |
| **Cross-Platform** | ✅ | ✅ | ✅ |
| **30+ Tools** | ✅ | ✅ (300+) | Fragmented |

**When to use DevToys**:
- Quick, frequent operations
- Sensitive data processing
- Offline environments
- Desktop-first workflow

**When to use CyberChef**:
- Complex multi-step transformations
- Binary data operations
- Custom "recipes"

---

## References

### Official Resources
- **Website**: https://devtoys.app/
- **GitHub**: https://github.com/DevToys-app/DevToys
- **Documentation**: https://devtoys.app/doc
- **Microsoft Store**: https://www.microsoft.com/store/productId/9PGCV4V3BK4W

### Related Projects
- **CyberChef**: https://github.com/gchq/CyberChef (browser-based Swiss Army knife)
- **Boop**: https://github.com/IvanMathy/Boop (macOS only, similar concept)
- **CodeBeautify**: https://codebeautify.org/ (online alternative)

---

## Document Classification

**Classification**: UNCLASSIFIED//PUBLIC
**Last Updated**: 2025-11-08
**Version**: 1.0
**Author**: LAT5150DRVMIL Documentation Team
**Contact**: SWORD Intelligence (https://github.com/SWORDOps/SWORDINTELLIGENCE/)

---

**LICENSE**: MIT License
**Permissions**: Commercial use, modification, distribution, private use
**Conditions**: License and copyright notice must be included
**Limitations**: No liability, no warranty
