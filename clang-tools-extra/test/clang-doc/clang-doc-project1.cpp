// RUN: rm -rf %t && mkdir -p %t/build %t/include %t/src %t/docs
// RUN: sed 's|$test_dir|%/t|g' %S/Inputs/clang-doc-project1/database_template.json > %t/build/compile_commands.json
// RUN: cp %S/Inputs/clang-doc-project1/*.h  %t/include
// RUN: cp %S/Inputs/clang-doc-project1/*.cpp %t/src
// RUN: cd %t
// RUN: clang-doc --format=html --executor=all-TUs --asset=%S/Inputs ./build/compile_commands.json
// RUN: FileCheck %s -input-file=%t/docs/index_json.js -check-prefix=CHECK-JSON-INDEX
// RUN: FileCheck %s -input-file=%t/docs/GlobalNamespace/Shape.html -check-prefix=CHECK-HTML-SHAPE
// RUN: FileCheck %s -input-file=%t/docs/GlobalNamespace/Calculator.html -check-prefix=CHECK-HTML-CALC
// RUN: FileCheck %s -input-file=%t/docs/GlobalNamespace/Rectangle.html -check-prefix=CHECK-HTML-RECTANGLE
// RUN: FileCheck %s -input-file=%t/docs/GlobalNamespace/Circle.html -check-prefix=CHECK-HTML-CIRCLE

// CHECK-JSON-INDEX: var JsonIndex = `
// CHECK-JSON-INDEX-NEXT: {
// CHECK-JSON-INDEX-NEXT:   "USR": "{{([0-9A-F]{40})}}",
// CHECK-JSON-INDEX-NEXT:   "Name": "",
// CHECK-JSON-INDEX-NEXT:   "RefType": "default",
// CHECK-JSON-INDEX-NEXT:   "Path": "",
// CHECK-JSON-INDEX-NEXT:   "Children": [
// CHECK-JSON-INDEX-NEXT:     {
// CHECK-JSON-INDEX-NEXT:       "USR": "{{([0-9A-F]{40})}}",
// CHECK-JSON-INDEX-NEXT:       "Name": "GlobalNamespace",
// CHECK-JSON-INDEX-NEXT:       "RefType": "namespace",
// CHECK-JSON-INDEX-NEXT:       "Path": "GlobalNamespace",
// CHECK-JSON-INDEX-NEXT:       "Children": [
// CHECK-JSON-INDEX-NEXT:         {
// CHECK-JSON-INDEX-NEXT:           "USR": "{{([0-9A-F]{40})}}",
// CHECK-JSON-INDEX-NEXT:           "Name": "Calculator",
// CHECK-JSON-INDEX-NEXT:           "RefType": "record",
// CHECK-JSON-INDEX-NEXT:           "Path": "GlobalNamespace",
// CHECK-JSON-INDEX-NEXT:           "Children": []
// CHECK-JSON-INDEX-NEXT:         },
// CHECK-JSON-INDEX-NEXT:         {
// CHECK-JSON-INDEX-NEXT:           "USR": "{{([0-9A-F]{40})}}",
// CHECK-JSON-INDEX-NEXT:           "Name": "Circle",
// CHECK-JSON-INDEX-NEXT:           "RefType": "record",
// CHECK-JSON-INDEX-NEXT:           "Path": "GlobalNamespace",
// CHECK-JSON-INDEX-NEXT:           "Children": []
// CHECK-JSON-INDEX-NEXT:         },
// CHECK-JSON-INDEX-NEXT:         {
// CHECK-JSON-INDEX-NEXT:           "USR": "{{([0-9A-F]{40})}}",
// CHECK-JSON-INDEX-NEXT:           "Name": "Rectangle",
// CHECK-JSON-INDEX-NEXT:           "RefType": "record",
// CHECK-JSON-INDEX-NEXT:           "Path": "GlobalNamespace",
// CHECK-JSON-INDEX-NEXT:           "Children": []
// CHECK-JSON-INDEX-NEXT:         },
// CHECK-JSON-INDEX-NEXT:         {
// CHECK-JSON-INDEX-NEXT:           "USR": "{{([0-9A-F]{40})}}",
// CHECK-JSON-INDEX-NEXT:           "Name": "Shape",
// CHECK-JSON-INDEX-NEXT:           "RefType": "record",
// CHECK-JSON-INDEX-NEXT:           "Path": "GlobalNamespace",
// CHECK-JSON-INDEX-NEXT:           "Children": []
// CHECK-JSON-INDEX-NEXT:         }
// CHECK-JSON-INDEX-NEXT:       ]
// CHECK-JSON-INDEX-NEXT:     }
// CHECK-JSON-INDEX-NEXT:   ]
// CHECK-JSON-INDEX-NEXT: }`;

// CHECK-HTML-SHAPE: <!DOCTYPE html>
// CHECK-HTML-SHAPE-NEXT: <meta charset="utf-8"/>
// CHECK-HTML-SHAPE-NEXT: <title>class Shape</title>
// CHECK-HTML-SHAPE-NEXT: <link rel="stylesheet" href="../clang-doc-default-stylesheet.css"/>
// CHECK-HTML-SHAPE-NEXT: <script src="../index.js"></script>
// CHECK-HTML-SHAPE-NEXT: <script src="../index_json.js"></script>
// CHECK-HTML-SHAPE-NEXT: <header id="project-title"></header>
// CHECK-HTML-SHAPE-NEXT: <main>
// CHECK-HTML-SHAPE-NEXT:   <div id="sidebar-left" path="GlobalNamespace" class="col-xs-6 col-sm-3 col-md-2 sidebar sidebar-offcanvas-left"></div>
// CHECK-HTML-SHAPE-NEXT:   <div id="main-content" class="col-xs-12 col-sm-9 col-md-8 main-content">
// CHECK-HTML-SHAPE-NEXT:     <h1>class Shape</h1>
// CHECK-HTML-SHAPE-NEXT:     <p>Defined at line 8 of file {{.*}}Shape.h</p>
// CHECK-HTML-SHAPE-NEXT:     <div>
// CHECK-HTML-SHAPE-NEXT:       <div>
// CHECK-HTML-SHAPE-NEXT:         <p> Provides a common interface for different types of shapes.</p>
// CHECK-HTML-SHAPE-NEXT:       </div>
// CHECK-HTML-SHAPE-NEXT:     </div>
// CHECK-HTML-SHAPE-NEXT:     <h2 id="Functions">Functions</h2>
// CHECK-HTML-SHAPE-NEXT:     <div>
// CHECK-HTML-SHAPE-NEXT:       <h3 id="{{([0-9A-F]{40})}}">~Shape</h3>
// CHECK-HTML-SHAPE-NEXT:       <p>public void ~Shape()</p>
// CHECK-HTML-SHAPE-NEXT:       <p>Defined at line 13 of file {{.*}}Shape.h</p>
// CHECK-HTML-SHAPE-NEXT:       <div>
// CHECK-HTML-SHAPE-NEXT:         <div></div>
// CHECK-HTML-SHAPE-NEXT:       </div>
// CHECK-HTML-SHAPE-NEXT:       <h3 id="{{([0-9A-F]{40})}}">area</h3>
// CHECK-HTML-SHAPE-NEXT:       <p>public double area()</p>
// CHECK-HTML-SHAPE-NEXT:       <div>
// CHECK-HTML-SHAPE-NEXT:         <div></div>
// CHECK-HTML-SHAPE-NEXT:       </div>
// CHECK-HTML-SHAPE-NEXT:       <h3 id="{{([0-9A-F]{40})}}">perimeter</h3>
// CHECK-HTML-SHAPE-NEXT:       <p>public double perimeter()</p>
// CHECK-HTML-SHAPE-NEXT:       <div>
// CHECK-HTML-SHAPE-NEXT:         <div></div>
// CHECK-HTML-SHAPE-NEXT:       </div>
// CHECK-HTML-SHAPE-NEXT:     </div>
// CHECK-HTML-SHAPE-NEXT:   </div>
// CHECK-HTML-SHAPE-NEXT:   <div id="sidebar-right" class="col-xs-6 col-sm-6 col-md-2 sidebar sidebar-offcanvas-right">
// CHECK-HTML-SHAPE-NEXT:     <ol>
// CHECK-HTML-SHAPE-NEXT:       <li>
// CHECK-HTML-SHAPE-NEXT:         <span>
// CHECK-HTML-SHAPE-NEXT:           <a href="#Functions">Functions</a>
// CHECK-HTML-SHAPE-NEXT:         </span>
// CHECK-HTML-SHAPE-NEXT:         <ul>
// CHECK-HTML-SHAPE-NEXT:           <li>
// CHECK-HTML-SHAPE-NEXT:             <span>
// CHECK-HTML-SHAPE-NEXT:               <a href="#{{([0-9A-F]{40})}}">~Shape</a>
// CHECK-HTML-SHAPE-NEXT:             </span>
// CHECK-HTML-SHAPE-NEXT:           </li>
// CHECK-HTML-SHAPE-NEXT:           <li>
// CHECK-HTML-SHAPE-NEXT:             <span>
// CHECK-HTML-SHAPE-NEXT:               <a href="#{{([0-9A-F]{40})}}">area</a>
// CHECK-HTML-SHAPE-NEXT:             </span>
// CHECK-HTML-SHAPE-NEXT:           </li>
// CHECK-HTML-SHAPE-NEXT:           <li>
// CHECK-HTML-SHAPE-NEXT:             <span>
// CHECK-HTML-SHAPE-NEXT:               <a href="#{{([0-9A-F]{40})}}">perimeter</a>
// CHECK-HTML-SHAPE-NEXT:             </span>
// CHECK-HTML-SHAPE-NEXT:           </li>
// CHECK-HTML-SHAPE-NEXT:         </ul>
// CHECK-HTML-SHAPE-NEXT:       </li>
// CHECK-HTML-SHAPE-NEXT:     </ol>
// CHECK-HTML-SHAPE-NEXT:   </div>
// CHECK-HTML-SHAPE-NEXT: </main>

// CHECK-HTML-CALC: <!DOCTYPE html>
// CHECK-HTML-CALC-NEXT: <meta charset="utf-8"/>
// CHECK-HTML-CALC-NEXT: <title>class Calculator</title>
// CHECK-HTML-CALC-NEXT: <link rel="stylesheet" href="{{.*}}clang-doc-default-stylesheet.css"/>
// CHECK-HTML-CALC-NEXT: <script src="{{.*}}index.js"></script>
// CHECK-HTML-CALC-NEXT: <script src="{{.*}}index_json.js"></script>
// CHECK-HTML-CALC-NEXT: <header id="project-title"></header>
// CHECK-HTML-CALC-NEXT: <main>
// CHECK-HTML-CALC-NEXT:   <div id="sidebar-left" path="GlobalNamespace" class="col-xs-6 col-sm-3 col-md-2 sidebar sidebar-offcanvas-left"></div>
// CHECK-HTML-CALC-NEXT:   <div id="main-content" class="col-xs-12 col-sm-9 col-md-8 main-content">
// CHECK-HTML-CALC-NEXT:     <h1>class Calculator</h1>
// CHECK-HTML-CALC-NEXT:     <p>Defined at line 8 of file {{.*}}Calculator.h</p>
// CHECK-HTML-CALC-NEXT:     <div>
// CHECK-HTML-CALC-NEXT:       <div>
// CHECK-HTML-CALC-NEXT:         <p> Provides basic arithmetic operations.</p>
// CHECK-HTML-CALC-NEXT:       </div>
// CHECK-HTML-CALC-NEXT:     </div>
// CHECK-HTML-CALC-NEXT:     <h2 id="Functions">Functions</h2>
// CHECK-HTML-CALC-NEXT:     <div>
// CHECK-HTML-CALC-NEXT:       <h3 id="{{([0-9A-F]{40})}}">add</h3>
// CHECK-HTML-CALC-NEXT:       <p>public int add(int a, int b)</p>
// CHECK-HTML-CALC-NEXT:       <p>Defined at line 4 of file {{.*}}Calculator.cpp</p>
// CHECK-HTML-CALC-NEXT:       <div>
// CHECK-HTML-CALC-NEXT:         <div></div>
// CHECK-HTML-CALC-NEXT:       </div>
// CHECK-HTML-CALC-NEXT:       <h3 id="{{([0-9A-F]{40})}}">subtract</h3>
// CHECK-HTML-CALC-NEXT:       <p>public int subtract(int a, int b)</p>
// CHECK-HTML-CALC-NEXT:       <p>Defined at line 8 of file {{.*}}Calculator.cpp</p>
// CHECK-HTML-CALC-NEXT:       <div>
// CHECK-HTML-CALC-NEXT:         <div></div>
// CHECK-HTML-CALC-NEXT:       </div>
// CHECK-HTML-CALC-NEXT:       <h3 id="{{([0-9A-F]{40})}}">multiply</h3>
// CHECK-HTML-CALC-NEXT:       <p>public int multiply(int a, int b)</p>
// CHECK-HTML-CALC-NEXT:       <p>Defined at line 12 of file {{.*}}Calculator.cpp</p>
// CHECK-HTML-CALC-NEXT:       <div>
// CHECK-HTML-CALC-NEXT:         <div></div>
// CHECK-HTML-CALC-NEXT:       </div>
// CHECK-HTML-CALC-NEXT:       <h3 id="{{([0-9A-F]{40})}}">divide</h3>
// CHECK-HTML-CALC-NEXT:       <p>public double divide(int a, int b)</p>
// CHECK-HTML-CALC-NEXT:       <p>Defined at line 16 of file {{.*}}Calculator.cpp</p>
// CHECK-HTML-CALC-NEXT:       <div>
// CHECK-HTML-CALC-NEXT:         <div></div>
// CHECK-HTML-CALC-NEXT:       </div>
// CHECK-HTML-CALC-NEXT:     </div>
// CHECK-HTML-CALC-NEXT:   </div>
// CHECK-HTML-CALC-NEXT:   <div id="sidebar-right" class="col-xs-6 col-sm-6 col-md-2 sidebar sidebar-offcanvas-right">
// CHECK-HTML-CALC-NEXT:     <ol>
// CHECK-HTML-CALC-NEXT:       <li>
// CHECK-HTML-CALC-NEXT:         <span>
// CHECK-HTML-CALC-NEXT:           <a href="#Functions">Functions</a>
// CHECK-HTML-CALC-NEXT:         </span>
// CHECK-HTML-CALC-NEXT:         <ul>
// CHECK-HTML-CALC-NEXT:           <li>
// CHECK-HTML-CALC-NEXT:             <span>
// CHECK-HTML-CALC-NEXT:               <a href="#{{([0-9A-F]{40})}}">add</a>
// CHECK-HTML-CALC-NEXT:             </span>
// CHECK-HTML-CALC-NEXT:           </li>
// CHECK-HTML-CALC-NEXT:           <li>
// CHECK-HTML-CALC-NEXT:             <span>
// CHECK-HTML-CALC-NEXT:               <a href="#{{([0-9A-F]{40})}}">subtract</a>
// CHECK-HTML-CALC-NEXT:             </span>
// CHECK-HTML-CALC-NEXT:           </li>
// CHECK-HTML-CALC-NEXT:           <li>
// CHECK-HTML-CALC-NEXT:             <span>
// CHECK-HTML-CALC-NEXT:               <a href="#{{([0-9A-F]{40})}}">multiply</a>
// CHECK-HTML-CALC-NEXT:             </span>
// CHECK-HTML-CALC-NEXT:           </li>
// CHECK-HTML-CALC-NEXT:           <li>
// CHECK-HTML-CALC-NEXT:             <span>
// CHECK-HTML-CALC-NEXT:               <a href="#{{([0-9A-F]{40})}}">divide</a>
// CHECK-HTML-CALC-NEXT:             </span>
// CHECK-HTML-CALC-NEXT:           </li>
// CHECK-HTML-CALC-NEXT:         </ul>
// CHECK-HTML-CALC-NEXT:       </li>
// CHECK-HTML-CALC-NEXT:     </ol>
// CHECK-HTML-CALC-NEXT:   </div>
// CHECK-HTML-CALC-NEXT: </main>

// CHECK-HTML-RECTANGLE: <!DOCTYPE html>
// CHECK-HTML-RECTANGLE-NEXT: <meta charset="utf-8"/>
// CHECK-HTML-RECTANGLE-NEXT: <title>class Rectangle</title>
// CHECK-HTML-RECTANGLE-NEXT: <link rel="stylesheet" href="{{.*}}clang-doc-default-stylesheet.css"/>
// CHECK-HTML-RECTANGLE-NEXT: <script src="{{.*}}index.js"></script>
// CHECK-HTML-RECTANGLE-NEXT: <script src="{{.*}}index_json.js"></script>
// CHECK-HTML-RECTANGLE-NEXT: <header id="project-title"></header>
// CHECK-HTML-RECTANGLE-NEXT: <main>
// CHECK-HTML-RECTANGLE-NEXT:   <div id="sidebar-left" path="GlobalNamespace" class="col-xs-6 col-sm-3 col-md-2 sidebar sidebar-offcanvas-left"></div>
// CHECK-HTML-RECTANGLE-NEXT:   <div id="main-content" class="col-xs-12 col-sm-9 col-md-8 main-content">
// CHECK-HTML-RECTANGLE-NEXT:     <h1>class Rectangle</h1>
// CHECK-HTML-RECTANGLE-NEXT:     <p>Defined at line 10 of file {{.*}}Rectangle.h</p>
// CHECK-HTML-RECTANGLE-NEXT:     <div>
// CHECK-HTML-RECTANGLE-NEXT:       <div>
// CHECK-HTML-RECTANGLE-NEXT:         <p> Represents a rectangle with a given width and height.</p>
// CHECK-HTML-RECTANGLE-NEXT:       </div>
// CHECK-HTML-RECTANGLE-NEXT:     </div>
// CHECK-HTML-RECTANGLE-NEXT:     <p>
// CHECK-HTML-RECTANGLE-NEXT:       Inherits from
// CHECK-HTML-RECTANGLE-NEXT:       <a href="Shape.html">Shape</a>
// CHECK-HTML-RECTANGLE-NEXT:     </p>
// CHECK-HTML-RECTANGLE-NEXT:     <h2 id="Members">Members</h2>
// CHECK-HTML-RECTANGLE-NEXT:     <ul>
// CHECK-HTML-RECTANGLE-NEXT:       <li>private double width_</li>
// CHECK-HTML-RECTANGLE-NEXT:       <li>private double height_</li>
// CHECK-HTML-RECTANGLE-NEXT:     </ul>
// CHECK-HTML-RECTANGLE-NEXT:     <h2 id="Functions">Functions</h2>
// CHECK-HTML-RECTANGLE-NEXT:     <div>
// CHECK-HTML-RECTANGLE-NEXT:       <h3 id="{{([0-9A-F]{40})}}">Rectangle</h3>
// CHECK-HTML-RECTANGLE-NEXT:       <p>public void Rectangle(double width, double height)</p>
// CHECK-HTML-RECTANGLE-NEXT:       <p>Defined at line 3 of file {{.*}}Rectangle.cpp</p>
// CHECK-HTML-RECTANGLE-NEXT:       <div>
// CHECK-HTML-RECTANGLE-NEXT:         <div></div>
// CHECK-HTML-RECTANGLE-NEXT:       </div>
// CHECK-HTML-RECTANGLE-NEXT:       <h3 id="{{([0-9A-F]{40})}}">area</h3>
// CHECK-HTML-RECTANGLE-NEXT:       <p>public double area()</p>
// CHECK-HTML-RECTANGLE-NEXT:       <p>Defined at line 6 of file {{.*}}Rectangle.cpp</p>
// CHECK-HTML-RECTANGLE-NEXT:       <div>
// CHECK-HTML-RECTANGLE-NEXT:         <div></div>
// CHECK-HTML-RECTANGLE-NEXT:       </div>
// CHECK-HTML-RECTANGLE-NEXT:       <h3 id="{{([0-9A-F]{40})}}">perimeter</h3>
// CHECK-HTML-RECTANGLE-NEXT:       <p>public double perimeter()</p>
// CHECK-HTML-RECTANGLE-NEXT:       <p>Defined at line 10 of file {{.*}}Rectangle.cpp</p>
// CHECK-HTML-RECTANGLE-NEXT:       <div>
// CHECK-HTML-RECTANGLE-NEXT:         <div></div>
// CHECK-HTML-RECTANGLE-NEXT:       </div>
// CHECK-HTML-RECTANGLE-NEXT:     </div>
// CHECK-HTML-RECTANGLE-NEXT:   </div>
// CHECK-HTML-RECTANGLE-NEXT:   <div id="sidebar-right" class="col-xs-6 col-sm-6 col-md-2 sidebar sidebar-offcanvas-right">
// CHECK-HTML-RECTANGLE-NEXT:     <ol>
// CHECK-HTML-RECTANGLE-NEXT:       <li>
// CHECK-HTML-RECTANGLE-NEXT:         <span>
// CHECK-HTML-RECTANGLE-NEXT:           <a href="#Members">Members</a>
// CHECK-HTML-RECTANGLE-NEXT:         </span>
// CHECK-HTML-RECTANGLE-NEXT:       </li>
// CHECK-HTML-RECTANGLE-NEXT:       <li>
// CHECK-HTML-RECTANGLE-NEXT:         <span>
// CHECK-HTML-RECTANGLE-NEXT:           <a href="#Functions">Functions</a>
// CHECK-HTML-RECTANGLE-NEXT:         </span>
// CHECK-HTML-RECTANGLE-NEXT:         <ul>
// CHECK-HTML-RECTANGLE-NEXT:           <li>
// CHECK-HTML-RECTANGLE-NEXT:             <span>
// CHECK-HTML-RECTANGLE-NEXT:               <a href="#{{([0-9A-F]{40})}}">Rectangle</a>
// CHECK-HTML-RECTANGLE-NEXT:             </span>
// CHECK-HTML-RECTANGLE-NEXT:           </li>
// CHECK-HTML-RECTANGLE-NEXT:           <li>
// CHECK-HTML-RECTANGLE-NEXT:             <span>
// CHECK-HTML-RECTANGLE-NEXT:               <a href="#{{([0-9A-F]{40})}}">area</a>
// CHECK-HTML-RECTANGLE-NEXT:             </span>
// CHECK-HTML-RECTANGLE-NEXT:           </li>
// CHECK-HTML-RECTANGLE-NEXT:           <li>
// CHECK-HTML-RECTANGLE-NEXT:             <span>
// CHECK-HTML-RECTANGLE-NEXT:               <a href="#{{([0-9A-F]{40})}}">perimeter</a>
// CHECK-HTML-RECTANGLE-NEXT:             </span>
// CHECK-HTML-RECTANGLE-NEXT:           </li>
// CHECK-HTML-RECTANGLE-NEXT:         </ul>
// CHECK-HTML-RECTANGLE-NEXT:       </li>
// CHECK-HTML-RECTANGLE-NEXT:     </ol>
// CHECK-HTML-RECTANGLE-NEXT:   </div>
// CHECK-HTML-RECTANGLE-NEXT: </main>

// CHECK-HTML-CIRCLE: <!DOCTYPE html>
// CHECK-HTML-CIRCLE-NEXT: <meta charset="utf-8"/>
// CHECK-HTML-CIRCLE-NEXT: <title>class Circle</title>
// CHECK-HTML-CIRCLE-NEXT: <link rel="stylesheet" href="{{.*}}clang-doc-default-stylesheet.css"/>
// CHECK-HTML-CIRCLE-NEXT: <script src="{{.*}}index.js"></script>
// CHECK-HTML-CIRCLE-NEXT: <script src="{{.*}}index_json.js"></script>
// CHECK-HTML-CIRCLE-NEXT: <header id="project-title"></header>
// CHECK-HTML-CIRCLE-NEXT: <main>
// CHECK-HTML-CIRCLE-NEXT:   <div id="sidebar-left" path="GlobalNamespace" class="col-xs-6 col-sm-3 col-md-2 sidebar sidebar-offcanvas-left"></div>
// CHECK-HTML-CIRCLE-NEXT:   <div id="main-content" class="col-xs-12 col-sm-9 col-md-8 main-content">
// CHECK-HTML-CIRCLE-NEXT:     <h1>class Circle</h1>
// CHECK-HTML-CIRCLE-NEXT:     <p>Defined at line 10 of file {{.*}}Circle.h</p>
// CHECK-HTML-CIRCLE-NEXT:     <div>
// CHECK-HTML-CIRCLE-NEXT:       <div>
// CHECK-HTML-CIRCLE-NEXT:         <p> Represents a circle with a given radius.</p>
// CHECK-HTML-CIRCLE-NEXT:       </div>
// CHECK-HTML-CIRCLE-NEXT:     </div>
// CHECK-HTML-CIRCLE-NEXT:     <p>
// CHECK-HTML-CIRCLE-NEXT:       Inherits from
// CHECK-HTML-CIRCLE-NEXT:       <a href="Shape.html">Shape</a>
// CHECK-HTML-CIRCLE-NEXT:     </p>
// CHECK-HTML-CIRCLE-NEXT:     <h2 id="Members">Members</h2>
// CHECK-HTML-CIRCLE-NEXT:     <ul>
// CHECK-HTML-CIRCLE-NEXT:       <li>private double radius_</li>
// CHECK-HTML-CIRCLE-NEXT:     </ul>
// CHECK-HTML-CIRCLE-NEXT:     <h2 id="Functions">Functions</h2>
// CHECK-HTML-CIRCLE-NEXT:     <div>
// CHECK-HTML-CIRCLE-NEXT:       <h3 id="{{([0-9A-F]{40})}}">Circle</h3>
// CHECK-HTML-CIRCLE-NEXT:       <p>public void Circle(double radius)</p>
// CHECK-HTML-CIRCLE-NEXT:       <p>Defined at line 3 of file {{.*}}Circle.cpp</p>
// CHECK-HTML-CIRCLE-NEXT:       <div>
// CHECK-HTML-CIRCLE-NEXT:         <div></div>
// CHECK-HTML-CIRCLE-NEXT:       </div>
// CHECK-HTML-CIRCLE-NEXT:       <h3 id="{{([0-9A-F]{40})}}">area</h3>
// CHECK-HTML-CIRCLE-NEXT:       <p>public double area()</p>
// CHECK-HTML-CIRCLE-NEXT:       <p>Defined at line 5 of file {{.*}}Circle.cpp</p>
// CHECK-HTML-CIRCLE-NEXT:       <div>
// CHECK-HTML-CIRCLE-NEXT:         <div></div>
// CHECK-HTML-CIRCLE-NEXT:       </div>
// CHECK-HTML-CIRCLE-NEXT:       <h3 id="{{([0-9A-F]{40})}}">perimeter</h3>
// CHECK-HTML-CIRCLE-NEXT:       <p>public double perimeter()</p>
// CHECK-HTML-CIRCLE-NEXT:       <p>Defined at line 9 of file {{.*}}Circle.cpp</p>
// CHECK-HTML-CIRCLE-NEXT:       <div>
// CHECK-HTML-CIRCLE-NEXT:         <div></div>
// CHECK-HTML-CIRCLE-NEXT:       </div>
// CHECK-HTML-CIRCLE-NEXT:     </div>
// CHECK-HTML-CIRCLE-NEXT:   </div>
// CHECK-HTML-CIRCLE-NEXT:   <div id="sidebar-right" class="col-xs-6 col-sm-6 col-md-2 sidebar sidebar-offcanvas-right">
// CHECK-HTML-CIRCLE-NEXT:     <ol>
// CHECK-HTML-CIRCLE-NEXT:       <li>
// CHECK-HTML-CIRCLE-NEXT:         <span>
// CHECK-HTML-CIRCLE-NEXT:           <a href="#Members">Members</a>
// CHECK-HTML-CIRCLE-NEXT:         </span>
// CHECK-HTML-CIRCLE-NEXT:       </li>
// CHECK-HTML-CIRCLE-NEXT:       <li>
// CHECK-HTML-CIRCLE-NEXT:         <span>
// CHECK-HTML-CIRCLE-NEXT:           <a href="#Functions">Functions</a>
// CHECK-HTML-CIRCLE-NEXT:         </span>
// CHECK-HTML-CIRCLE-NEXT:         <ul>
// CHECK-HTML-CIRCLE-NEXT:           <li>
// CHECK-HTML-CIRCLE-NEXT:             <span>
// CHECK-HTML-CIRCLE-NEXT:               <a href="#{{([0-9A-F]{40})}}">Circle</a>
// CHECK-HTML-CIRCLE-NEXT:             </span>
// CHECK-HTML-CIRCLE-NEXT:           </li>
// CHECK-HTML-CIRCLE-NEXT:           <li>
// CHECK-HTML-CIRCLE-NEXT:             <span>
// CHECK-HTML-CIRCLE-NEXT:               <a href="#{{([0-9A-F]{40})}}">area</a>
// CHECK-HTML-CIRCLE-NEXT:             </span>
// CHECK-HTML-CIRCLE-NEXT:           </li>
// CHECK-HTML-CIRCLE-NEXT:           <li>
// CHECK-HTML-CIRCLE-NEXT:             <span>
// CHECK-HTML-CIRCLE-NEXT:               <a href="#{{([0-9A-F]{40})}}">perimeter</a>
// CHECK-HTML-CIRCLE-NEXT:             </span>
// CHECK-HTML-CIRCLE-NEXT:           </li>
// CHECK-HTML-CIRCLE-NEXT:         </ul>
// CHECK-HTML-CIRCLE-NEXT:       </li>
// CHECK-HTML-CIRCLE-NEXT:     </ol>
// CHECK-HTML-CIRCLE-NEXT:   </div>
// CHECK-HTML-CIRCLE-NEXT: </main>