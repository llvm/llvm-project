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

// HTML-INDEX: <!DOCTYPE html>
// HTML-INDEX-NEXT: <meta charset="utf-8"/>
// HTML-INDEX-NEXT: <title>Global Namespace</title>
// HTML-INDEX-NEXT: <link rel="stylesheet" href="..{{[\/]}}clang-doc-default-stylesheet.css"/>
// HTML-INDEX-NEXT: <script src="..{{[\/]}}index_json.js"></script>
// HTML-INDEX-NEXT: <script src="..{{[\/]}}index.js"></script>
// HTML-INDEX-NEXT: <header id="project-title"></header>
// HTML-INDEX-NEXT: <main>
// HTML-INDEX-NEXT:   <div id="sidebar-left" path="GlobalNamespace" class="col-xs-6 col-sm-3 col-md-2 sidebar sidebar-offcanvas-left"></div>
// HTML-INDEX-NEXT:   <div id="main-content" class="col-xs-12 col-sm-9 col-md-8 main-content">
// HTML-INDEX-NEXT:     <h1>Global Namespace</h1>
// HTML-INDEX-NEXT:     <h2 id="Enums">Enums</h2>
// HTML-INDEX-NEXT:     <div>
// HTML-INDEX-NEXT:       <h3 id="{{([0-9A-F]{40})}}">enum Color</h3>
// HTML-INDEX-NEXT:       <ul>
// HTML-INDEX-NEXT:         <li>Red</li>
// HTML-INDEX-NEXT:         <li>Green</li>
// HTML-INDEX-NEXT:         <li>Blue</li>
// HTML-INDEX-NEXT:       </ul>
// HTML-INDEX-NEXT:       <p>Defined at line 11 of file {{.*}}clang-tools-extra{{[\/]}}test{{[\/]}}clang-doc{{[\/]}}enum.cpp</p>
// HTML-INDEX-NEXT:       <div>
// HTML-INDEX-NEXT:         <div></div>
// HTML-INDEX-NEXT:       </div>
// HTML-INDEX-NEXT:     </div>
// HTML-INDEX-NEXT:   </div>
// HTML-INDEX-NEXT:   <div id="sidebar-right" class="col-xs-6 col-sm-6 col-md-2 sidebar sidebar-offcanvas-right">
// HTML-INDEX-NEXT:     <ol>
// HTML-INDEX-NEXT:       <li>
// HTML-INDEX-NEXT:         <span>
// HTML-INDEX-NEXT:           <a href="#Enums">Enums</a>
// HTML-INDEX-NEXT:         </span>
// HTML-INDEX-NEXT:         <ul>
// HTML-INDEX-NEXT:           <li>
// HTML-INDEX-NEXT:             <span>
// HTML-INDEX-NEXT:               <a href="#{{([0-9A-F]{40})}}">Color</a>
// HTML-INDEX-NEXT:             </span>
// HTML-INDEX-NEXT:           </li>
// HTML-INDEX-NEXT:         </ul>
// HTML-INDEX-NEXT:       </li>
// HTML-INDEX-NEXT:     </ol>
// HTML-INDEX-NEXT:   </div>
// HTML-INDEX-NEXT: </main>

// MD-INDEX: # Global Namespace
// MD-INDEX: ## Enums
// MD-INDEX: | enum Color |
// MD-INDEX: --
// MD-INDEX: | Red |
// MD-INDEX: | Green |
// MD-INDEX: | Blue |
// MD-INDEX: *Defined at {{.*}}clang-tools-extra{{[\/]}}test{{[\/]}}clang-doc{{[\/]}}enum.cpp#11*
// MD-INDEX: **brief** For specifying RGB colors