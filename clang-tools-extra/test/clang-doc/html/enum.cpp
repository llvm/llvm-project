// RUN: rm -rf %t && mkdir -p %t
// RUN: clang-doc --format=html --doxygen --output=%t --executor=standalone %S/../Inputs/enum.cpp
// RUN: FileCheck %s < %t/html/GlobalNamespace/index.html --check-prefix=HTML-INDEX
// RUN: FileCheck %s < %t/html/GlobalNamespace/_ZTV7Animals.html --check-prefix=HTML-ANIMAL
// RUN: FileCheck %s < %t/html/GlobalNamespace/_ZTV15FilePermissions.html --check-prefix=HTML-PERM
// RUN: FileCheck %s < %t/html/Vehicles/index.html --check-prefix=HTML-VEHICLES

// HTML-INDEX-LABEL:  <div id="{{([0-9A-F]{40})}}" class="delimiter-container">
// HTML-INDEX-NEXT:     <div>
// HTML-INDEX-NEXT:       <pre><code class="language-cpp code-clang-doc">enum Color</code></pre>
// HTML-INDEX-NEXT:     </div>
// HTML-INDEX-NEXT:     <table class="table-wrapper">
// HTML-INDEX-NEXT:         <tbody>
// HTML-INDEX-NEXT:             <tr>
// HTML-INDEX-NEXT:                 <th>Name</th>
// HTML-INDEX-NEXT:                 <th>Value</th>
// HTML-INDEX-NEXT:                 <th>Comments</th>
// HTML-INDEX-NEXT:             </tr>
// HTML-INDEX-NEXT:             <tr>
// HTML-INDEX-NEXT:                 <td>Red</td>
// HTML-INDEX-NEXT:                 <td>0</td>
// HTML-INDEX-NEXT:                 <td>
// HTML-INDEX-NEXT:                   <p class="paragraph-container">
// HTML-INDEX-NEXT:                       Comment 1<br>
// HTML-INDEX-NEXT:                   </p>
// HTML-INDEX-NEXT:                 </td>
// HTML-INDEX-NEXT:             </tr>
// HTML-INDEX-NEXT:             <tr>
// HTML-INDEX-NEXT:                 <td>Green</td>
// HTML-INDEX-NEXT:                 <td>1</td>
// HTML-INDEX-NEXT:                 <td>
// HTML-INDEX-NEXT:                   <p class="paragraph-container">
// HTML-INDEX-NEXT:                       Comment 2<br>
// HTML-INDEX-NEXT:                   </p>
// HTML-INDEX-NEXT:                 </td>
// HTML-INDEX-NEXT:             </tr>
// HTML-INDEX-NEXT:             <tr>
// HTML-INDEX-NEXT:                 <td>Blue</td>
// HTML-INDEX-NEXT:                 <td>2</td>
// HTML-INDEX-NEXT:                 <td>
// HTML-INDEX-NEXT:                   <p class="paragraph-container">
// HTML-INDEX-NEXT:                       Comment 3<br>
// HTML-INDEX-NEXT:                   </p>
// HTML-INDEX-NEXT:                 </td>
// HTML-INDEX-NEXT:             </tr>
// HTML-INDEX-NEXT:         </tbody>
// HTML-INDEX-NEXT:     </table>
// HTML-INDEX-NEXT:     <div class="doc-card">
// HTML-INDEX-NEXT:       <div class="nested-delimiter-container">
// HTML-INDEX-NEXT:           <p>For specifying RGB colors</p>
// HTML-INDEX-NEXT:       </div>
// HTML-INDEX-NEXT:     </div>
// HTML-INDEX:        </div>

// HTML-INDEX-LABEL:  <div id="{{([0-9A-F]{40})}}" class="delimiter-container">
// HTML-INDEX-NEXT:     <div>
// HTML-INDEX-NEXT:       <pre><code class="language-cpp code-clang-doc">enum class Shapes</code></pre>
// HTML-INDEX-NEXT:     </div>
// HTML-INDEX-NEXT:     <table class="table-wrapper">
// HTML-INDEX-NEXT:         <tbody>
// HTML-INDEX-NEXT:             <tr>
// HTML-INDEX-NEXT:                 <th>Name</th>
// HTML-INDEX-NEXT:                 <th>Value</th>
// HTML-INDEX-NEXT:                 <th>Comments</th>
// HTML-INDEX-NEXT:             </tr>
// HTML-INDEX-NEXT:             <tr>
// HTML-INDEX-NEXT:                 <td>Circle</td>
// HTML-INDEX-NEXT:                 <td>0</td>
// HTML-INDEX-NEXT:                 <td>
// HTML-INDEX-NEXT:                   <p class="paragraph-container">
// HTML-INDEX-NEXT:                       Comment 1<br>
// HTML-INDEX-NEXT:                   </p>
// HTML-INDEX-NEXT:                 </td>
// HTML-INDEX-NEXT:             </tr>
// HTML-INDEX-NEXT:             <tr>
// HTML-INDEX-NEXT:                 <td>Rectangle</td>
// HTML-INDEX-NEXT:                 <td>1</td>
// HTML-INDEX-NEXT:                 <td>
// HTML-INDEX-NEXT:                   <p class="paragraph-container">
// HTML-INDEX-NEXT:                       Comment 2<br>
// HTML-INDEX-NEXT:                   </p>
// HTML-INDEX-NEXT:                 </td>
// HTML-INDEX-NEXT:             </tr>
// HTML-INDEX-NEXT:             <tr>
// HTML-INDEX-NEXT:                 <td>Triangle</td>
// HTML-INDEX-NEXT:                 <td>2</td>
// HTML-INDEX-NEXT:                 <td>
// HTML-INDEX-NEXT:                   <p class="paragraph-container">
// HTML-INDEX-NEXT:                       Comment 3<br>
// HTML-INDEX-NEXT:                   </p>
// HTML-INDEX-NEXT:                 </td>
// HTML-INDEX-NEXT:             </tr>
// HTML-INDEX-NEXT:         </tbody>
// HTML-INDEX-NEXT:     </table>
// HTML-INDEX-NEXT:     <div class="doc-card">
// HTML-INDEX-NEXT:       <div class="nested-delimiter-container">
// HTML-INDEX-NEXT:           <p>Shape Types</p>
// HTML-INDEX-NEXT:       </div>
// HTML-INDEX-NEXT:     </div>
// HTML-INDEX:        </div>

// HTML-INDEX-LABEL:   <div id="{{([0-9A-F]{40})}}" class="delimiter-container">
// HTML-INDEX-NEXT:     <div>
// HTML-INDEX-NEXT:       <pre><code class="language-cpp code-clang-doc">enum Size : uint8_t</code></pre>
// HTML-INDEX-NEXT:     </div>
// HTML-INDEX-NEXT:     <table class="table-wrapper">
// HTML-INDEX-NEXT:         <tbody>
// HTML-INDEX-NEXT:             <tr>
// HTML-INDEX-NEXT:                 <th>Name</th>
// HTML-INDEX-NEXT:                 <th>Value</th>
// HTML-INDEX-NEXT:                 <th>Comments</th>
// HTML-INDEX-NEXT:             </tr>
// HTML-INDEX-NEXT:             <tr>
// HTML-INDEX-NEXT:                 <td>Small</td>
// HTML-INDEX-NEXT:                 <td>0</td>
// HTML-INDEX-NEXT:                 <td>
// HTML-INDEX-NEXT:                   <p class="paragraph-container">
// HTML-INDEX-NEXT:                       A pearl.<br>
// HTML-INDEX-NEXT:                       Pearls are quite small.<br>
// HTML-INDEX-NEXT:                   </p>
// HTML-INDEX-NEXT:                   <p class="paragraph-container">
// HTML-INDEX-NEXT:                       Pearls are used in jewelry.<br>
// HTML-INDEX-NEXT:                   </p>
// HTML-INDEX-NEXT:                 </td>
// HTML-INDEX-NEXT:             </tr>
// HTML-INDEX-NEXT:             <tr>
// HTML-INDEX-NEXT:                 <td>Medium</td>
// HTML-INDEX-NEXT:                 <td>1</td>
// HTML-INDEX-NEXT:                 <td>
// HTML-INDEX-NEXT:                   <p class="paragraph-container">A tennis ball.</p>
// HTML-INDEX-NEXT:                 </td>
// HTML-INDEX-NEXT:             </tr>
// HTML-INDEX-NEXT:             <tr>
// HTML-INDEX-NEXT:                 <td>Large</td>
// HTML-INDEX-NEXT:                 <td>2</td>
// HTML-INDEX-NEXT:                 <td>
// HTML-INDEX-NEXT:                   <p class="paragraph-container">
// HTML-INDEX-NEXT:                       A football.<br>
// HTML-INDEX-NEXT:                   </p>
// HTML-INDEX-NEXT:                 </td>
// HTML-INDEX-NEXT:             </tr>
// HTML-INDEX-NEXT:         </tbody>
// HTML-INDEX-NEXT:     </table>
// HTML-INDEX-NEXT:     <div class="doc-card">
// HTML-INDEX-NEXT:       <div class="nested-delimiter-container">
// HTML-INDEX-NEXT:           <p>Specify the size</p>
// HTML-INDEX-NEXT:       </div>
// HTML-INDEX-NEXT:     </div>
// HTML-INDEX:        </div>

// HTML-INDEX-LABEL:  <div id="{{([0-9A-F]{40})}}" class="delimiter-container">
// HTML-INDEX-NEXT:     <div>
// HTML-INDEX-NEXT:       <pre><code class="language-cpp code-clang-doc">enum (unnamed) : long long</code></pre>
// HTML-INDEX-NEXT:     </div>
// HTML-INDEX-NEXT:     <table class="table-wrapper">
// HTML-INDEX-NEXT:         <tbody>
// HTML-INDEX-NEXT:             <tr>
// HTML-INDEX-NEXT:                 <th>Name</th>
// HTML-INDEX-NEXT:                 <th>Value</th>
// HTML-INDEX-NEXT:                 <th>Comments</th>
// HTML-INDEX-NEXT:             </tr>
// HTML-INDEX-NEXT:             <tr>
// HTML-INDEX-NEXT:                 <td>BigVal</td>
// HTML-INDEX-NEXT:                 <td>999999999999</td>
// HTML-INDEX-NEXT:                 <td>
// HTML-INDEX-NEXT:                   <p class="paragraph-container">
// HTML-INDEX-NEXT:                       A very large value<br>
// HTML-INDEX-NEXT:                   </p>
// HTML-INDEX-NEXT:                 </td>
// HTML-INDEX-NEXT:             </tr>
// HTML-INDEX-NEXT:         </tbody>
// HTML-INDEX-NEXT:     </table>
// HTML-INDEX-NEXT:     <div class="doc-card">
// HTML-INDEX-NEXT:       <div class="nested-delimiter-container">
// HTML-INDEX-NEXT:           <p>Very long number</p>
// HTML-INDEX-NEXT:       </div>
// HTML-INDEX-NEXT:     </div>
// HTML-INDEX:        </div>

// HTML-INDEX-LABEL:  <div id="{{([0-9A-F]{40})}}" class="delimiter-container">
// HTML-INDEX-NEXT:     <div>
// HTML-INDEX-NEXT:       <pre><code class="language-cpp code-clang-doc">enum ColorUserSpecified</code></pre>
// HTML-INDEX-NEXT:     </div>
// HTML-INDEX-NEXT:     <table class="table-wrapper">
// HTML-INDEX-NEXT:         <tbody>
// HTML-INDEX-NEXT:             <tr>
// HTML-INDEX-NEXT:                 <th>Name</th>
// HTML-INDEX-NEXT:                 <th>Value</th>
// HTML-INDEX-NEXT:             </tr>
// HTML-INDEX-NEXT:             <tr>
// HTML-INDEX-NEXT:                 <td>RedUserSpecified</td>
// HTML-INDEX-NEXT:                 <td>&#39;A&#39;</td>
// HTML-INDEX-NEXT:             </tr>
// HTML-INDEX-NEXT:             <tr>
// HTML-INDEX-NEXT:                 <td>GreenUserSpecified</td>
// HTML-INDEX-NEXT:                 <td>2</td>
// HTML-INDEX-NEXT:             </tr>
// HTML-INDEX-NEXT:             <tr>
// HTML-INDEX-NEXT:                 <td>BlueUserSpecified</td>
// HTML-INDEX-NEXT:                 <td>&#39;C&#39;</td>
// HTML-INDEX-NEXT:             </tr>
// HTML-INDEX-NEXT:         </tbody>
// HTML-INDEX-NEXT:     </table>
// HTML-INDEX:        </div>

// HTML-PERM-LABEL:  <section id="Enums" class="section-container">
// HTML-PERM-NEXT:     <h2>Enumerations</h2>
// HTML-PERM-NEXT:     <div id="{{([0-9A-F]{40})}}" class="delimiter-container">
// HTML-PERM-NEXT:       <div>
// HTML-PERM-NEXT:         <pre><code class="language-cpp code-clang-doc">enum (unnamed)</code></pre>
// HTML-PERM-NEXT:       </div>
// HTML-PERM-NEXT:       <table class="table-wrapper">
// HTML-PERM-NEXT:           <tbody>
// HTML-PERM-NEXT:               <tr>
// HTML-PERM-NEXT:                   <th>Name</th>
// HTML-PERM-NEXT:                   <th>Value</th>
// HTML-PERM-NEXT:                   <th>Comments</th>
// HTML-PERM-NEXT:               </tr>
// HTML-PERM-NEXT:               <tr>
// HTML-PERM-NEXT:                   <td>Read</td>
// HTML-PERM-NEXT:                   <td>1</td>
// HTML-PERM-NEXT:                   <td>
// HTML-PERM-NEXT:                     <p class="paragraph-container">
// HTML-PERM-NEXT:                         Permission to READ r<br>
// HTML-PERM-NEXT:                     </p>
// HTML-PERM-NEXT:                   </td>
// HTML-PERM-NEXT:               </tr>
// HTML-PERM-NEXT:               <tr>
// HTML-PERM-NEXT:                   <td>Write</td>
// HTML-PERM-NEXT:                   <td>2</td>
// HTML-PERM-NEXT:                  <td>
// HTML-PERM-NEXT:                     <p class="paragraph-container">
// HTML-PERM-NEXT:                         Permission to WRITE w<br>
// HTML-PERM-NEXT:                     </p>
// HTML-PERM-NEXT:                   </td>
// HTML-PERM-NEXT:               </tr>
// HTML-PERM-NEXT:               <tr>
// HTML-PERM-NEXT:                   <td>Execute</td>
// HTML-PERM-NEXT:                   <td>4</td>
// HTML-PERM-NEXT:                   <td>
// HTML-PERM-NEXT:                     <p class="paragraph-container">
// HTML-PERM-NEXT:                         Permission to EXECUTE x<br>
// HTML-PERM-NEXT:                     </p>
// HTML-PERM-NEXT:                   </td>
// HTML-PERM-NEXT:               </tr>
// HTML-PERM-NEXT:           </tbody>
// HTML-PERM-NEXT:       </table>
// HTML-PERM-NEXT:       <div class="doc-card">
// HTML-PERM-NEXT:         <div class="nested-delimiter-container">
// HTML-PERM-NEXT:             <p>File permission flags</p>
// HTML-PERM-NEXT:         </div>
// HTML-PERM:        </section>

// HTML-ANIMAL-LABEL:   <section id="Enums" class="section-container">
// HTML-ANIMAL-NEXT:      <h2>Enumerations</h2>
// HTML-ANIMAL-NEXT:      <div id="{{([0-9A-F]{40})}}" class="delimiter-container">
// HTML-ANIMAL-NEXT:         <div>
// HTML-ANIMAL-NEXT:           <pre><code class="language-cpp code-clang-doc">enum AnimalType</code></pre>
// HTML-ANIMAL-NEXT:         </div>
// HTML-ANIMAL-NEXT:         <table class="table-wrapper">
// HTML-ANIMAL-NEXT:             <tbody>
// HTML-ANIMAL-NEXT:                 <tr>
// HTML-ANIMAL-NEXT:                     <th>Name</th>
// HTML-ANIMAL-NEXT:                     <th>Value</th>
// HTML-ANIMAL-NEXT:                     <th>Comments</th>
// HTML-ANIMAL-NEXT:                 </tr>
// HTML-ANIMAL-NEXT:                 <tr>
// HTML-ANIMAL-NEXT:                     <td>Dog</td>
// HTML-ANIMAL-NEXT:                     <td>0</td>
// HTML-ANIMAL-NEXT:                     <td>
// HTML-ANIMAL-NEXT:                       <p class="paragraph-container">
// HTML-ANIMAL-NEXT:                           Man&#39;s best friend<br>
// HTML-ANIMAL-NEXT:                       </p>
// HTML-ANIMAL-NEXT:                     </td>
// HTML-ANIMAL-NEXT:                 </tr>
// HTML-ANIMAL-NEXT:                 <tr>
// HTML-ANIMAL-NEXT:                     <td>Cat</td>
// HTML-ANIMAL-NEXT:                     <td>1</td>
// HTML-ANIMAL-NEXT:                     <td>
// HTML-ANIMAL-NEXT:                       <p class="paragraph-container">
// HTML-ANIMAL-NEXT:                           Man&#39;s other best friend<br>
// HTML-ANIMAL-NEXT:                       </p>
// HTML-ANIMAL-NEXT:                     </td>
// HTML-ANIMAL-NEXT:                 </tr>
// HTML-ANIMAL-NEXT:                 <tr>
// HTML-ANIMAL-NEXT:                     <td>Iguana</td>
// HTML-ANIMAL-NEXT:                     <td>2</td>
// HTML-ANIMAL-NEXT:                     <td>
// HTML-ANIMAL-NEXT:                       <p class="paragraph-container">
// HTML-ANIMAL-NEXT:                           A lizard<br>
// HTML-ANIMAL-NEXT:                       </p>
// HTML-ANIMAL-NEXT:                     </td>
// HTML-ANIMAL-NEXT:                 </tr>
// HTML-ANIMAL-NEXT:             </tbody>
// HTML-ANIMAL-NEXT:         </table>
// HTML-ANIMAL-NEXT:         <div class="doc-card">
// HTML-ANIMAL-NEXT:             <div class="nested-delimiter-container">
// HTML-ANIMAL-NEXT:                 <p>specify what animal the class is</p>
// HTML-ANIMAL-NEXT:             </div>
// HTML-ANIMAL-NEXT:         </div>
// HTML-ANIMAL:           </div>
// HTML-ANIMAL-NEXT:    </section>

// HTML-VEHICLES-LABEL:   <div id="{{([0-9A-F]{40})}}" class="delimiter-container">
// HTML-VEHICLES-NEXT:      <div>
// HTML-VEHICLES-NEXT:       <pre><code class="language-cpp code-clang-doc">enum Car</code></pre>
// HTML-VEHICLES-NEXT:      </div>
// HTML-VEHICLES-NEXT:      <table class="table-wrapper">
// HTML-VEHICLES-NEXT:          <tbody>
// HTML-VEHICLES-NEXT:              <tr>
// HTML-VEHICLES-NEXT:                  <th>Name</th>
// HTML-VEHICLES-NEXT:                  <th>Value</th>
// HTML-VEHICLES-NEXT:                  <th>Comments</th>
// HTML-VEHICLES-NEXT:              </tr>
// HTML-VEHICLES-NEXT:              <tr>
// HTML-VEHICLES-NEXT:                  <td>Sedan</td>
// HTML-VEHICLES-NEXT:                  <td>0</td>
// HTML-VEHICLES-NEXT:                  <td>
// HTML-VEHICLES-NEXT:                    <p class="paragraph-container">
// HTML-VEHICLES-NEXT:                        Comment 1<br>
// HTML-VEHICLES-NEXT:                    </p>
// HTML-VEHICLES-NEXT:                  </td>
// HTML-VEHICLES-NEXT:              </tr>
// HTML-VEHICLES-NEXT:              <tr>
// HTML-VEHICLES-NEXT:                  <td>SUV</td>
// HTML-VEHICLES-NEXT:                  <td>1</td>
// HTML-VEHICLES-NEXT:                  <td>
// HTML-VEHICLES-NEXT:                    <p class="paragraph-container">
// HTML-VEHICLES-NEXT:                        Comment 2<br>
// HTML-VEHICLES-NEXT:                    </p>
// HTML-VEHICLES-NEXT:                  </td>
// HTML-VEHICLES-NEXT:              </tr>
// HTML-VEHICLES-NEXT:              <tr>
// HTML-VEHICLES-NEXT:                  <td>Pickup</td>
// HTML-VEHICLES-NEXT:                  <td>2</td>
// HTML-VEHICLES-NEXT:                  <td> -- </td>
// HTML-VEHICLES-NEXT:              </tr>
// HTML-VEHICLES-NEXT:              <tr>
// HTML-VEHICLES-NEXT:                  <td>Hatchback</td>
// HTML-VEHICLES-NEXT:                  <td>3</td>
// HTML-VEHICLES-NEXT:                  <td>
// HTML-VEHICLES-NEXT:                    <p class="paragraph-container">
// HTML-VEHICLES-NEXT:                        Comment 4<br>
// HTML-VEHICLES-NEXT:                    </p>
// HTML-VEHICLES-NEXT:                  </td>
// HTML-VEHICLES-NEXT:              </tr>
// HTML-VEHICLES-NEXT:          </tbody>
// HTML-VEHICLES-NEXT:      </table>
// HTML-VEHICLES-NEXT:      <div class="doc-card">
// HTML-VEHICLES-NEXT:        <div class="nested-delimiter-container">
// HTML-VEHICLES-NEXT:           <p>specify type of car</p>
// HTML-VEHICLES-NEXT:        </div>
// HTML-VEHICLES-NEXT:      </div>
// HTML-VEHICLES:         </div>
