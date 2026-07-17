// Force-included (via `clang -include`) at the top of every Boost translation
// unit by the profiles Boost build (.github/workflows/profiles-boost-build.yml)
// so that the std::init profile is enforced across Boost without editing any
// Boost source.
//
// This must contain ONLY the enforcement empty-declaration: a
// [[profiles::enforce(...)]] attribute has to precede every other top-level
// declaration in the translation unit (P3589R2 s1.1.1), which `-include`
// guarantees by inserting this file before the main file's contents.
[[profiles::enforce(std::init)]];
