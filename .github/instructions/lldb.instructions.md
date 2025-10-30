---
applyTo: lldb/**/*
---

When reviewing code, focus on:

## Language, Libraries & Standards

- Target C++17 and avoid vendor-specific extensions.
- For Python scripts, follow PEP 8.
- Prefer standard library or LLVM support libraries instead of reinventing data structures.

## Comments & Documentation

- Each source file should include the standard LLVM file header.
- Header files must have proper header guards.
- Non-trivial classes and public methods should have Doxygen documentation.
- Use `//` or `///` comments normally; avoid block comments unless necessary.

## Language & Compiler Issues

- Write portable code; wrap non-portable code in interfaces.
- Do not use RTTI or exceptions.
- Prefer C++-style casts over C-style casts.
- Avoid static constructors or global objects with heavy initialization.
- Use `class` or `struct` consistently; `struct` only for all-public data.

## Headers & Library Layering

- Include order: module header → local/private headers → project headers → system headers.
- Headers must compile standalone (include all dependencies).
- Maintain proper library layering; avoid circular dependencies.
- Include minimally; use forward declarations where possible.
- Keep internal headers private to modules.
- Use full namespace qualifiers for out-of-line definitions.

## Control Flow & Structure

- Prefer early exits over deep nesting.
- Avoid `else` after `return`, `continue`, `break`, or `goto`.
- Encapsulate loops that compute predicates into helper functions.

## Naming

- LLDB's code style differs from LLVM's coding style.
- Variables are `snake_case`.
- Functions and methods are `UpperCamelCase`.
- Static, global and member variables have `s_`, `g_` and `m_` prefixes respectively.

## General Guidelines

- Use `assert` liberally; prefer `llvm_unreachable` for unreachable states.
- Avoid `using namespace std;` in headers.
- Ensure at least one out-of-line virtual method per class with virtuals.
- For `switch` on enums, omit `default` to catch missing cases.
- Prefer range-based `for` loops.
- Capture `end()` outside loops if not using range-based iteration.
- Use LLVM’s `raw_ostream` instead of `<iostream>`.
- Avoid `std::endl`; use `\n` unless flushing is needed.
- Methods in class definitions are already inline—don’t add `inline`.

## Microscopic Details

- Preserve existing style in modified code.
- Prefer pre-increment (`++i`) when value is unused.
- Omit braces for single-statement `if`, `else`, `while`, `for` unless needed.

## Review Style

- Be specific and actionable in feedback.
- Explain the "why" behind recommendations.
- Link back to the LLVM Coding Standars: https://llvm.org/docs/CodingStandards.html.
- Ask clarifying questions when code intent is unclear.

Remember that these standards are **guidelines**. Always prioritize consistency
with the style that is already being used by the surrounding code.
