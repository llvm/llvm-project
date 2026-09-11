```{title} clang-tidy - misc-confusable-identifiers
```

# misc-confusable-identifiers

Warn about confusable identifiers, i.e. identifiers that are visually close to
each other, but use different Unicode characters. This detects a potential
attack described in [CVE-2021-42574](https://www.cve.org/CVERecord?id=CVE-2021-42574).

Example:

```text
int fo; // Initial character is U+0066 (LATIN SMALL LETTER F).
int 𝐟o; // Initial character is U+1D41F (MATHEMATICAL BOLD SMALL F) not U+0066 (LATIN SMALL LETTER F).
```

