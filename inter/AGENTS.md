# Inter Development Rules

## Dialect Metadata

- Define operation metadata structurally in ODS. Do not attach declared
  operation fields with raw `setAttr` calls.
- Access ODS fields through generated operation accessors. Do not look up
  operation fields by string name in handwritten C++.
- Do not introduce constants for internal ODS field names. A named string
  constant does not make a stringly-typed operation contract structural.
- When multiple operation kinds expose the same semantic property, define a
  dialect operation interface and consume that interface outside the dialect.
- Interfaces must expose both reads and mutations needed by consumers. Do not
  bypass an interface with raw attribute access for updates.
- Keep serialized metadata names centralized in the owning dialect when no
  generated structural accessor can exist.
