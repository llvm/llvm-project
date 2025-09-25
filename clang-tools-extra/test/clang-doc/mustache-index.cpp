// RUN: rm -rf %t && mkdir -p %t
// RUN: clang-doc --format=mustache --output=%t --executor=standalone %s 
// RUN: FileCheck %s < %t/html/index.html

enum Color {
  RED,
  BLUE,
  GREEN
};

class Foo;

// CHECK:       <li class="sidebar-section">
// CHECK-NEXT:      <a class="sidebar-item" href="#Enums">Enums</a>
// CHECK-NEXT:  </li>
// CHECK-NEXT:  <ul>
// CHECK-NEXT:      <li class="sidebar-item-container">
// CHECK-NEXT:          <a class="sidebar-item" href="#{{[0-9A-F]*}}">Color</a>
// CHECK-NEXT:      </li>
// CHECK-NEXT:  </ul>
// CHECK:           <li class="sidebar-section">
// CHECK-NEXT:          <a class="sidebar-item" href="#Classes">Inner Classes</a>
// CHECK-NEXT:      </li>
// CHECK-NEXT:  <ul>
// CHECK-NEXT:      <li class="sidebar-item-container">
// CHECK-NEXT:          <a class="sidebar-item" href="#{{[0-9A-F]*}}">Foo</a>
// CHECK-NEXT:      </li>
// CHECK-NEXT:  </ul>

// CHECK:       <section id="Enums" class="section-container">
// CHECK-NEXT:      <h2>Enumerations</h2>
// CHECK-NEXT:      <div>
// CHECK-NEXT:          <div id="{{[0-9A-F]*}}" class="delimiter-container">
// CHECK-NEXT:              <div>
// CHECK-NEXT:                  <pre>
// CHECK-NEXT:                      <code class="language-cpp code-clang-doc">
// CHECK-NEXT:                          enum Color
// CHECK-NEXT:                      </code>
// CHECK-NEXT:                  </pre>
// CHECK-NEXT:              </div>
// CHECK-NEXT:              <table class="table-wrapper">
// CHECK-NEXT:                  <tbody>
// CHECK-NEXT:                  <tr>
// CHECK-NEXT:                      <th>Name</th>
// CHECK-NEXT:                      <th>Value</th>
// CHECK:                       </tr>
// CHECK-NEXT:                      <tr>
// CHECK-NEXT:                          <td>RED</td>
// CHECK-NEXT:                          <td>0</td>
// CHECK:                           </tr>
// CHECK-NEXT:                      <tr>
// CHECK-NEXT:                          <td>BLUE</td>
// CHECK-NEXT:                          <td>1</td>
// CHECK:                           </tr>
// CHECK-NEXT:                      <tr>
// CHECK-NEXT:                          <td>GREEN</td>
// CHECK-NEXT:                          <td>2</td>
// CHECK:                           </tr>
// CHECK-NEXT:                  </tbody>
// CHECK-NEXT:              </table>
// CHECK-NEXT:              <div>
// CHECK-NEXT:                  Defined at line 5 of file {{.*}}mustache-index.cpp
// CHECK-NEXT:              </div>
// CHECK-NEXT:          </div>
// CHECK-NEXT:      </div>
// CHECK-NEXT:  </section>

// CHECK:       <section id="Classes" class="section-container">
// CHECK-NEXT:      <h2>Inner Classes</h2>
// CHECK-NEXT:      <ul class="class-container">
// CHECK-NEXT:          <li id="{{[0-9A-F]*}}" style="max-height: 40px;">
// CHECK-NEXT:              <a href="_ZTV3Foo.html">
// CHECK-NEXT:                  <pre>
// CHECK-NEXT:                      <code class="language-cpp code-clang-doc">class Foo</code>
// CHECK-NEXT:                  </pre>
// CHECK-NEXT:              </a>
// CHECK-NEXT:          </li>
// CHECK-NEXT:      </ul>
// CHECK-NEXT:  </section>
