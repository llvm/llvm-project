// RUN: rm -rf %t && mkdir -p %t
// RUN: clang-doc --extra-arg -std=c++20 --output=%t --doxygen --format=html --executor=standalone %s
// RUN: FileCheck %s < %t/json/GlobalNamespace/index.json
// RUN: FileCheck %s < %t/html/GlobalNamespace/index.html -check-prefix=CHECK-HTML

/// \brief Concept for an incrementable value
///
/// \tparam T A value that can be incremented.
template<typename T> concept Incrementable = requires (T a) {
  a++;
};

/// \brief Concept for a decrementable value
///
/// \tparam T A value that can be decremented
template<typename T> concept Decrementable = requires (T a) {
  a--;
};

/// \brief Concept for a pre-incrementable value
///
/// \tparam T A value that can be pre-incremented
template<typename T> concept PreIncrementable = requires (T a) {
  ++a;
};

/// \brief Concept for a -pre-decrementable value
///
/// \tparam T A value that can be pre-decremented
template<typename T> concept PreDecrementable = requires (T a) {
  --a;
};

template<typename T> requires Incrementable<T> && Decrementable<T> void One();

template<typename T> requires (Incrementable<T> && Decrementable<T>) void Two();

template<typename T> requires (Incrementable<T> && Decrementable<T>) || (PreIncrementable<T> && PreDecrementable<T>) void Three();

template<typename T> requires (Incrementable<T> && Decrementable<T>) || PreIncrementable<T> void Four();

// CHECK:         "Name": "One",
// CHECK:         "Template": {
// CHECK-NEXT:      "Constraints": [
// CHECK-NEXT:        {
// CHECK-NEXT:          "Expression": "Incrementable<T>",
// CHECK-NEXT:          "Name": "Incrementable",
// CHECK-NEXT:          "QualName": "Incrementable",
// CHECK-NEXT:          "USR": "{{[0-9A-F]*}}"
// CHECK-NEXT:        },
// CHECK-NEXT:        {
// CHECK-NEXT:          "End": true,
// CHECK-NEXT:          "Expression": "Decrementable<T>",
// CHECK-NEXT:          "Name": "Decrementable",
// CHECK-NEXT:          "QualName": "Decrementable",
// CHECK-NEXT:          "USR": "{{[0-9A-F]*}}"
// CHECK-NEXT:        }
// CHECK-NEXT:      ],
// CHECK:         "Name": "Two",
// CHECK:         "Template": {
// CHECK-NEXT:      "Constraints": [
// CHECK-NEXT:        {
// CHECK-NEXT:          "Expression": "Incrementable<T>",
// CHECK-NEXT:          "Name": "Incrementable",
// CHECK-NEXT:          "QualName": "Incrementable",
// CHECK-NEXT:          "USR": "{{[0-9A-F]*}}"
// CHECK-NEXT:        },
// CHECK-NEXT:        {
// CHECK-NEXT:          "End": true,
// CHECK-NEXT:          "Expression": "Decrementable<T>",
// CHECK-NEXT:          "Name": "Decrementable",
// CHECK-NEXT:          "QualName": "Decrementable",
// CHECK-NEXT:          "USR": "{{[0-9A-F]*}}"
// CHECK-NEXT:        }
// CHECK-NEXT:      ],
// CHECK:         "Name": "Three",
// CHECK:         "Template": {
// CHECK-NEXT:      "Constraints": [
// CHECK-NEXT:        {
// CHECK-NEXT:          "Expression": "Incrementable<T>",
// CHECK-NEXT:          "Name": "Incrementable",
// CHECK-NEXT:          "QualName": "Incrementable",
// CHECK-NEXT:          "USR": "{{[0-9A-F]*}}"
// CHECK-NEXT:        },
// CHECK-NEXT:        {
// CHECK-NEXT:          "Expression": "Decrementable<T>",
// CHECK-NEXT:          "Name": "Decrementable",
// CHECK-NEXT:          "QualName": "Decrementable",
// CHECK-NEXT:          "USR": "{{[0-9A-F]*}}"
// CHECK-NEXT:        },
// CHECK-NEXT:        {
// CHECK-NEXT:          "Expression": "PreIncrementable<T>",
// CHECK-NEXT:          "Name": "PreIncrementable",
// CHECK-NEXT:          "QualName": "PreIncrementable",
// CHECK-NEXT:          "USR": "{{[0-9A-F]*}}"
// CHECK-NEXT:        },
// CHECK-NEXT:        {
// CHECK-NEXT:          "End": true,
// CHECK-NEXT:          "Expression": "PreDecrementable<T>",
// CHECK-NEXT:          "Name": "PreDecrementable",
// CHECK-NEXT:          "QualName": "PreDecrementable",
// CHECK-NEXT:          "USR": "{{[0-9A-F]*}}"
// CHECK-NEXT:        }
// CHECK-NEXT:      ],
// CHECK:         "Name": "Four",
// CHECK:         "Template": {
// CHECK-NEXT:      "Constraints": [
// CHECK-NEXT:        {
// CHECK-NEXT:          "Expression": "Incrementable<T>",
// CHECK-NEXT:          "Name": "Incrementable",
// CHECK-NEXT:          "QualName": "Incrementable",
// CHECK-NEXT:          "USR": "{{[0-9A-F]*}}"
// CHECK-NEXT:        },
// CHECK-NEXT:        {
// CHECK-NEXT:          "Expression": "Decrementable<T>",
// CHECK-NEXT:          "Name": "Decrementable",
// CHECK-NEXT:          "QualName": "Decrementable",
// CHECK-NEXT:          "USR": "{{[0-9A-F]*}}"
// CHECK-NEXT:        },
// CHECK-NEXT:        {
// CHECK-NEXT:          "End": true,
// CHECK-NEXT:          "Expression": "PreIncrementable<T>",
// CHECK-NEXT:          "Name": "PreIncrementable",
// CHECK-NEXT:          "QualName": "PreIncrementable",
// CHECK-NEXT:          "USR": "{{[0-9A-F]*}}"
// CHECK-NEXT:        }
// CHECK-NEXT:      ],

// CHECK-HTML:          <a class="sidebar-item" href="#Concepts">Concepts</a>
// CHECK-HTML-NEXT: </summary>
// CHECK-HTML-NEXT: <ul>
// CHECK-HTML-NEXT:     <li class="sidebar-item-container">
// CHECK-HTML-NEXT:         <a class="sidebar-item" href="#{{([0-9A-F]{40})}}">Incrementable</a>
// CHECK-HTML-NEXT:     </li>
// CHECK-HTML-NEXT:     <li class="sidebar-item-container">
// CHECK-HTML-NEXT:         <a class="sidebar-item" href="#{{([0-9A-F]{40})}}">Decrementable</a>
// CHECK-HTML-NEXT:     </li>
// CHECK-HTML-NEXT:     <li class="sidebar-item-container">
// CHECK-HTML-NEXT:         <a class="sidebar-item" href="#{{([0-9A-F]{40})}}">PreIncrementable</a>
// CHECK-HTML-NEXT:     </li>
// CHECK-HTML-NEXT:     <li class="sidebar-item-container">
// CHECK-HTML-NEXT:         <a class="sidebar-item" href="#{{([0-9A-F]{40})}}">PreDecrementable</a>
// CHECK-HTML-NEXT:     </li>
// CHECK-HTML-NEXT: </ul>
// CHECK-HTML:      <section id="Concepts" class="section-container">
// CHECK-HTML-NEXT:     <h2>Concepts</h2>
// CHECK-HTML-NEXT:     <div id="{{([0-9A-F]{40})}}" class="delimiter-container">
// CHECK-HTML-NEXT:         <div>
// CHECK-HTML-NEXT:             <pre><code class="language-cpp code-clang-doc">template &lt;typename T&gt; Incrementable requires (T a) { a++; }</code></pre> 
// CHECK-HTML-NEXT:         </div>
// CHECK-HTML-NEXT:         <div class="nested-delimiter-container">
// CHECK-HTML-NEXT:             <p> Concept for an incrementable value</p>
// CHECK-HTML-NEXT:         </div>
// CHECK-HTML-NEXT:         <h3>Template Parameters</h3>
// CHECK-HTML-NEXT:         <div class="nested-delimiter-container">
// CHECK-HTML-NEXT:             <div>
// CHECK-HTML-NEXT:                 <b>T</b>   A value that can be incremented.
// CHECK-HTML-NEXT:             </div>
// CHECK-HTML-NEXT:         </div> 
// CHECK-HTML-NEXT:         <p>Defined at line [[@LINE-151]] of file {{.*}}compound-constraints.cpp</p>
// CHECK-HTML-NEXT:     </div>
// CHECK-HTML-NEXT:     <div id="{{([0-9A-F]{40})}}" class="delimiter-container">
// CHECK-HTML-NEXT:         <div>
// CHECK-HTML-NEXT:             <pre><code class="language-cpp code-clang-doc">template &lt;typename T&gt; Decrementable requires (T a) { a--; }</code></pre> 
// CHECK-HTML-NEXT:         </div>
// CHECK-HTML-NEXT:         <div class="nested-delimiter-container">
// CHECK-HTML-NEXT:             <p> Concept for a decrementable value</p>
// CHECK-HTML-NEXT:         </div>
// CHECK-HTML-NEXT:         <h3>Template Parameters</h3>
// CHECK-HTML-NEXT:         <div class="nested-delimiter-container">
// CHECK-HTML-NEXT:             <div>
// CHECK-HTML-NEXT:                 <b>T</b>   A value that can be decremented
// CHECK-HTML-NEXT:             </div>
// CHECK-HTML-NEXT:         </div> 
// CHECK-HTML-NEXT:         <p>Defined at line [[@LINE-159]] of file {{.*}}compound-constraints.cpp</p>
// CHECK-HTML-NEXT:     </div>
// CHECK-HTML-NEXT:     <div id="{{([0-9A-F]{40})}}" class="delimiter-container">
// CHECK-HTML-NEXT:         <div>
// CHECK-HTML-NEXT:             <pre><code class="language-cpp code-clang-doc">template &lt;typename T&gt; PreIncrementable requires (T a) { ++a; }</code></pre> 
// CHECK-HTML-NEXT:         </div>
// CHECK-HTML-NEXT:         <div class="nested-delimiter-container">
// CHECK-HTML-NEXT:             <p> Concept for a pre-incrementable value</p>
// CHECK-HTML-NEXT:         </div>
// CHECK-HTML-NEXT:         <h3>Template Parameters</h3>
// CHECK-HTML-NEXT:         <div class="nested-delimiter-container">
// CHECK-HTML-NEXT:             <div>
// CHECK-HTML-NEXT:                 <b>T</b>   A value that can be pre-incremented
// CHECK-HTML-NEXT:             </div>
// CHECK-HTML-NEXT:         </div> 
// CHECK-HTML-NEXT:         <p>Defined at line [[@LINE-167]] of file {{.*}}compound-constraints.cpp</p>
// CHECK-HTML-NEXT:     </div>
// CHECK-HTML-NEXT:     <div id="{{([0-9A-F]{40})}}" class="delimiter-container">
// CHECK-HTML-NEXT:         <div>
// CHECK-HTML-NEXT:             <pre><code class="language-cpp code-clang-doc">template &lt;typename T&gt; PreDecrementable requires (T a) { --a; }</code></pre> 
// CHECK-HTML-NEXT:         </div>
// CHECK-HTML-NEXT:         <div class="nested-delimiter-container">
// CHECK-HTML-NEXT:             <p> Concept for a -pre-decrementable value</p>
// CHECK-HTML-NEXT:         </div>
// CHECK-HTML-NEXT:         <h3>Template Parameters</h3>
// CHECK-HTML-NEXT:         <div class="nested-delimiter-container">
// CHECK-HTML-NEXT:             <div>
// CHECK-HTML-NEXT:                 <b>T</b>   A value that can be pre-decremented
// CHECK-HTML-NEXT:             </div>
// CHECK-HTML-NEXT:         </div> 
// CHECK-HTML-NEXT:         <p>Defined at line [[@LINE-175]] of file {{.*}}compound-constraints.cpp</p>
// CHECK-HTML-NEXT:     </div>
// CHECK-HTML-NEXT: </section>
