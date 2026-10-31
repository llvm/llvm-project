// RUN: rm -rf %t && mkdir -p %t
// RUN: clang-doc --pretty-json --doxygen --executor=standalone %S/../Inputs/templates.cpp -output=%t/docs --format=html
// RUN: cat %t/docs/html/GlobalNamespace/_ZTV5tuple.html | FileCheck %s --check-prefix=HTML-STRUCT
// RUN: cat %t/docs/html/GlobalNamespace/index.html | FileCheck %s --check-prefix=HTML

// HTML:        <pre><code class="language-cpp code-clang-doc">template &lt;class... T</code><code class="language-cpp code-clang-doc">&gt;</code></pre>
// HTML-NEXT:   <pre><code class="language-cpp code-clang-doc">void ParamPackFunction (T... args)</code></pre>

// HTML:           <pre><code class="language-cpp code-clang-doc">template &lt;typename T, int U = 1</code><code class="language-cpp code-clang-doc">&gt;</code></pre>
// HTML-NEXT:      <pre><code class="language-cpp code-clang-doc">void function (T x)</code></pre>
// HTML-NEXT:      <p>Defined at line 3 of file {{.*}}templates.cpp</p>
// HTML-NEXT:  </div>

// HTML:           <pre>
// HTML-SAME:        <code class="language-cpp code-clang-doc">template &lt;</code>
// HTML-SAME:        <span class="param-container">
// HTML-SAME:          <span class="param"><code class="language-cpp code-clang-doc">typename A, </code></span>
// HTML-SAME:          <span class="param"><code class="language-cpp code-clang-doc">typename B, </code></span>
// HTML-SAME:          <span class="param"><code class="language-cpp code-clang-doc">typename C, </code></span>
// HTML-SAME:          <span class="param"><code class="language-cpp code-clang-doc">typename D, </code></span>
// HTML-SAME:          <span class="param"><code class="language-cpp code-clang-doc">typename E</code></span>
// HTML-SAME:        </span>
// HTML-SAME:        <code class="language-cpp code-clang-doc">&gt;</code>
// HTML-SAME:      </pre>
// HTML-NEXT:      <pre>
// HTML-SAME:        <code class="language-cpp code-clang-doc">void longFunction (</code>
// HTML-SAME:        <span class="param-container">
// HTML-SAME:          <span class="param"><code class="language-cpp code-clang-doc">A</code> <code class="language-cpp code-clang-doc">a, </code></span>
// HTML-SAME:          <span class="param"><code class="language-cpp code-clang-doc">B</code> <code class="language-cpp code-clang-doc">b, </code></span>
// HTML-SAME:          <span class="param"><code class="language-cpp code-clang-doc">C</code> <code class="language-cpp code-clang-doc">c, </code></span>
// HTML-SAME:          <span class="param"><code class="language-cpp code-clang-doc">D</code> <code class="language-cpp code-clang-doc">d, </code></span>
// HTML-SAME:          <span class="param"><code class="language-cpp code-clang-doc">E</code> <code class="language-cpp code-clang-doc">e</code></span>
// HTML-SAME:        </span>
// HTML-SAME:        <code class="language-cpp code-clang-doc">)</code>
// HTML-SAME:      </pre>
// HTML-NEXT:      <p>Defined at line 6 of file {{.*}}templates.cpp</p>
// HTML-NEXT:  </div>

// HTML:           <pre><code class="language-cpp code-clang-doc">template &lt;</code><code class="language-cpp code-clang-doc">&gt;</code></pre>
// HTML-NEXT:      <pre><code class="language-cpp code-clang-doc">void function&lt;bool, 0&gt; (bool x)</code></pre>
// HTML-NEXT:      <p>Defined at line 8 of file {{.*}}templates.cpp</p>
// HTML-NEXT:  </div>

// HTML-STRUCT:        <section class="hero section-container">
// HTML-STRUCT-NEXT:       <pre><code class="language-cpp code-clang-doc">template &lt;typename... Tys&gt;</code></pre>
// HTML-STRUCT-NEXT:       <div class="hero__title">
// HTML-STRUCT-NEXT:           <h1 class="hero__title-large">struct tuple</h1>
// HTML-STRUCT-NEXT:           <p>Defined at line 13 of file {{.*}}templates.cpp</p>
// HTML-STRUCT-NEXT:           <div class="doc-card">
// HTML-STRUCT-NEXT:               <div class="nested-delimiter-container">
// HTML-STRUCT-NEXT:                   <p>A Tuple type</p>
// HTML-STRUCT-NEXT:               </div>
// HTML-STRUCT-NEXT:               <div class="nested-delimiter-container">
// HTML-STRUCT-NEXT:                   <p>Does Tuple things.</p>
// HTML-STRUCT-NEXT:               </div>
// HTML-STRUCT-NEXT:           </div>
// HTML-STRUCT-NEXT:       </div>
// HTML-STRUCT-NEXT:   </section>

// HTML:           <pre><code class="language-cpp code-clang-doc">tuple&lt;int, int, bool&gt; func_with_tuple_param (tuple&lt;int, int, bool&gt; t)</code></pre>
// HTML-NEXT:      <div class="doc-card">
// HTML-NEXT:          <div class="nested-delimiter-container">
// HTML-NEXT:              <p>A function with a tuple parameter</p>
// HTML-NEXT:          </div>
// HTML-NEXT:          <div class="nested-delimiter-container">
// HTML-NEXT:              <h3>Parameters</h3>
// HTML-NEXT:              <div>
// HTML-NEXT:                  <b>t</b>   The input to func_with_tuple_param
// HTML-NEXT:              </div>
// HTML-NEXT:          </div>
// HTML-NEXT:      </div>
// HTML-NEXT:      <p>Defined at line 18 of file {{.*}}templates.cpp</p>
// HTML-NEXT:  </div>
