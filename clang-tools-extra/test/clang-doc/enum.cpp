// RUN: rm -rf %t && mkdir -p %t
// RUN: clang-doc --format=html --doxygen --output=%t --executor=standalone %s
// RUN: clang-doc --format=md --doxygen --output=%t --executor=standalone %s
// RUN: FileCheck %s < %t/GlobalNamespace/index.html -check-prefix=HTML-INDEX
// RUN: FileCheck %s < %t/GlobalNamespace/index.md -check-prefix=MD-INDEX

/**
 * @brief For specifying RGB colors
 */
enum Color {
  Red,
  Green,
  Blue
};

// HTML-INDEX: <h1>Global Namespace</h1>
// HTML-INDEX: <h2 id="Enums">Enums</h2>
// HTML-INDEX: <h3 id="{{([0-9A-F]{40})}}">enum Color</h3>
// HTML-INDEX: <li>Red</li>
// HTML-INDEX: <li>Green</li>
// HTML-INDEX: <li>Blue</li>
// HTML-INDEX: <p>Defined at line 10 of file {{.*}}clang-tools-extra{{[\/]}}test{{[\/]}}clang-doc{{[\/]}}enum.cpp</p>

// MD-INDEX: # Global Namespace
// MD-INDEX: ## Enums
// MD-INDEX: | enum Color |
// MD-INDEX: --
// MD-INDEX: | Red |
// MD-INDEX: | Green |
// MD-INDEX: | Blue |
// MD-INDEX: *Defined at {{.*}}clang-tools-extra{{[\/]}}test{{[\/]}}clang-doc{{[\/]}}enum.cpp#10*
// MD-INDEX: **brief** For specifying RGB colors