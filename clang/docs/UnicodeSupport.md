# Unicode Support

Clang uses a subset of the [Unicode Character Database](https://www.unicode.org/ucd/)
(UCD) for identifiers, `\N{...}` named characters, diagnostic column width,
and simple case folding. The corresponding tables are generated from the UCD
into Clang and LLVM Support.

## Updating the Unicode Data

Replace `N.M.0` with the Unicode version being imported. Build the generators
from an LLVM build directory that has utilities enabled (`LLVM_BUILD_UTILS`,
the default) and libxml2 (`LLVM_ENABLE_LIBXML2`, also the default):

```bash
ninja -C <build> UnicodeCharSetsGenerator UnicodeNameMappingGenerator
```

Then run them from the `llvm-project` root as shown below.

### Character properties

`UnicodeCharSetsGenerator` writes the identifier character sets used by the
lexer (`XID_Start`, `XID_Continue`, and the mathematical compatibility notation
profile) and the printable / formatting / combining / East-Asian-width sets
used by LLVM Support.

Download the [flat XML UCD](https://www.unicode.org/Public/UCD/latest/ucdxml/)
(`ucd.nounihan.flat.zip` is enough):

```text
https://www.unicode.org/Public/N.M.0/ucdxml/ucd.nounihan.flat.zip
```

```bash
<build>/bin/UnicodeCharSetsGenerator ucd.nounihan.flat.xml \
    clang/lib/Lex/UnicodeCharSetsGenerated.cpp \
    llvm/lib/Support/UnicodeCharSetsGenerated.cpp
clang-format -i clang/lib/Lex/UnicodeCharSetsGenerated.cpp \
    llvm/lib/Support/UnicodeCharSetsGenerated.cpp
```

The C99 and C11 identifier tables and the whitespace table in
`clang/lib/Lex/UnicodeCharSets.h` are maintained by hand.

### Character names

`UnicodeNameMappingGenerator` writes the name-to-codepoint trie used by
`\N{...}` and `llvm::sys::unicode::nameToCodepointStrict` /
`nameToCodepointLooseMatching`.

From the [UCD zip](https://www.unicode.org/Public/UCD/latest/ucd/UCD.zip), take
`UnicodeData.txt` and `NameAliases.txt`:

```text
https://www.unicode.org/Public/N.M.0/ucd/UnicodeData.txt
https://www.unicode.org/Public/N.M.0/ucd/NameAliases.txt
```

```bash
<build>/bin/UnicodeNameMappingGenerator UnicodeData.txt NameAliases.txt \
    llvm/lib/Support/UnicodeNameToCodepointGenerated.cpp
clang-format -i llvm/lib/Support/UnicodeNameToCodepointGenerated.cpp
```

Algorithmically derived names (`CJK UNIFIED IDEOGRAPH-*`, and so on) are not
in those files. Update `GeneratedNamesDataTable` in
`llvm/lib/Support/UnicodeNameToCodepoint.cpp` from
`extracted/DerivedName.txt` in the same UCD zip.

### Case folding

`llvm/utils/unicode-case-fold.py` fetches `CaseFolding.txt` and writes
`llvm/lib/Support/UnicodeCaseFold.cpp`:

```bash
llvm/utils/unicode-case-fold.py \
    https://www.unicode.org/Public/N.M.0/ucd/CaseFolding.txt \
    > llvm/lib/Support/UnicodeCaseFold.cpp
```

After regenerating the tables, update
`llvm/unittests/Support/UnicodeTest.cpp` and `clang/test/Lexer/unicode.c` for
new characters and changed name ranges.
