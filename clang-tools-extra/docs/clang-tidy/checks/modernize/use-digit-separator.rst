.. title:: clang-tidy - modernize-use-digit-separator

modernize-use-digit-separator
=============================

The check looks for long integral constants and inserts the digits separator
(') appropriately. Groupings:
    - decimal integral constants, groups of 3 digits, e.g. int x = 1'000;
    - binary integral constants, groups of 4 digits, e.g. int x = 0b0010'0011;
    - octal integral constants, groups of 3 digits, e.g. int x = 0377'777;
    - hexadecimal integral constants, groups of 4 digits, e.g. unsigned long
    x = 0xffff'0000;
    - floating-point constants, group into 3 digits on either side of the
    decimal point, e.g. float x = 3'456.001'25f;
