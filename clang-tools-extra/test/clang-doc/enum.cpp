// RUN: clang-doc --format=html --doxygen --output=%t/docs --executor=standalone %s
// RUN: clang-doc --format=md --doxygen --output=%t/docs --executor=standalone %s
// RUN: FileCheck %s -input-file=%t/docs/GlobalNamespace/index.html -check-prefix=HTML-INDEX
// RUN: FileCheck %s -input-file=%t/docs/GlobalNamespace/index.md -check-prefix=MD-INDEX

/**
 * @brief For specifying RGB colors
 */
enum Color {
  Red, // Red enums
  Green, // Green enums
  Blue // Blue enums
};

// HTML-INDEX: <h1>Global Namespace</h1>
// HTML-INDEX: <h2 id="Enums">Enums</h2>
// HTML-INDEX: <div>
// HTML-INDEX:   <h3 id="{{([0-9A-F]{40})}}">enum Color</h3>
// HTML-INDEX:   <ul>
// HTML-INDEX:     <li>Red</li>
// HTML-INDEX:     <li>Green</li>
// HTML-INDEX:     <li>Blue</li>
// HTML-INDEX:   </ul>
// HTML-INDEX:   <p>Defined at line 11 of file {{.*}}clang-tools-extra{{[\/]}}test{{[\/]}}clang-doc{{[\/]}}enum.cpp</p>
// HTML-INDEX:   <div>
// HTML-INDEX:     <div></div>
// HTML-INDEX:   </div>
// HTML-INDEX: </div>

// MD-INDEX: # Global Namespace
// MD-INDEX: ## Enums
// MD-INDEX: | enum Color |
// MD-INDEX: --
// MD-INDEX: | Red |
// MD-INDEX: | Green |
// MD-INDEX: | Blue |
// MD-INDEX: *Defined at {{.*}}clang-tools-extra{{[\/]}}test{{[\/]}}clang-doc{{[\/]}}enum.cpp#11*
// MD-INDEX: **brief** For specifying RGB colors