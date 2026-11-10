// A stub stand-in for clang/include/clang/Basic/DiagnosticDocs.td. The emitter
// copies Intro to the top of the generated file verbatim, and no test checks
// its contents, so there is nothing to gain from mirroring the real banner,
// stylesheet, and role definitions here.

def GlobalDocumentation {
  code Intro =[{
## Diagnostic flags
}];
}
