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
- Non-trivial code should have comments explaining what it does and why. Avoid comments that explain how it does it at a micro level.

## Language & Compiler Issues

- Write portable code; wrap non-portable code in interfaces.
- Do not use RTTI or exceptions.
- Prefer C++-style casts over C-style casts.
- Do not use static constructors.
- Use `class` or `struct` consistently; `struct` only for all-public data.
- When then same class is declared or defined multiple times, make sure it's consistently done using either `class` or `struct`.

## Headers & Library Layering

- Include order: module header → local/private headers → project headers → system headers.
- Headers must compile standalone (include all dependencies).
- Maintain proper library layering; avoid circular dependencies.
- Include minimally; use forward declarations where possible.
- Keep internal headers private to modules.
- Use full namespace qualifiers for out-of-line definitions.

## Control Flow & Structure

- Prefer early exits over deep nesting.
- Do not use `else` after `return`, `continue`, `break`, or `goto`.
- Encapsulate loops that compute predicates into helper functions.

## Naming

- LLDB's code style differs from LLVM's coding style.
- Variables are `snake_case`.
- Functions and methods are `UpperCamelCase`.
- Static, global and member variables have `s_`, `g_` and `m_` prefixes respectively.

## General Guidelines

- Use `assert` liberally; prefer `llvm_unreachable` for unreachable states.
- Do not use `using namespace std;` in headers.
- Provide a virtual method anchor for classes defined in headers.
- Do not use default labels in fully covered switches over enumerations.
- Use range-based for loops wherever possible.
- Capture `end()` outside loops if not using range-based iteration.
- Including `<iostream>` is forbidded. Use LLVM’s `raw_ostream` instead.
- Don’t use `inline` when defining a function in a class definition.

## Microscopic Details

- Preserve existing style in modified code.
- Prefer pre-increment (`++i`) when value is unused.
- Use `private`, `protected`, or `public` keyword as appropriate to restrict class member visibility.
- Omit braces for single-statement `if`, `else`, `while`, `for` unless needed.

## Review Style

- Be specific and actionable in feedback.
- Explain the "why" behind recommendations.
- Link back to the LLVM Coding Standards: https://llvm.org/docs/CodingStandards.html.
- Ask clarifying questions when code intent is unclear.

Ignore formatting and assume that's handled by external tools like `clang-format` and `black`.
Remember that these standards are **guidelines**.
Always prioritize consistency with the style that is already being used by the surrounding code.
