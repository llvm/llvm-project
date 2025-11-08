// RUN: rm -rf %t && mkdir -p %t
// RUN: clang-doc --format=html --doxygen --output=%t --executor=standalone %s
// RUN: clang-doc --format=md --doxygen --output=%t --executor=standalone %s
// RUN: FileCheck %s < %t/html/GlobalNamespace/index.html --check-prefix=HTML-INDEX-LINE
// RUN: FileCheck %s < %t/html/GlobalNamespace/index.html --check-prefix=HTML-INDEX
// RUN: FileCheck %s < %t/html/GlobalNamespace/_ZTV7Animals.html --check-prefix=HTML-ANIMAL-LINE
// RUN: FileCheck %s < %t/html/GlobalNamespace/_ZTV7Animals.html --check-prefix=HTML-ANIMAL
// RUN: FileCheck %s < %t/html/Vehicles/index.html --check-prefix=HTML-VEHICLES-LINE
// RUN: FileCheck %s < %t/html/Vehicles/index.html --check-prefix=HTML-VEHICLES
// RUN: FileCheck %s < %t/GlobalNamespace/index.md --check-prefix=MD-INDEX-LINE
// RUN: FileCheck %s < %t/GlobalNamespace/index.md --check-prefix=MD-INDEX
// RUN: FileCheck %s < %t/GlobalNamespace/Animals.md --check-prefix=MD-ANIMAL-LINE
// RUN: FileCheck %s < %t/GlobalNamespace/Animals.md --check-prefix=MD-ANIMAL
// RUN: FileCheck %s < %t/Vehicles/index.md --check-prefix=MD-VEHICLES-LINE
// RUN: FileCheck %s < %t/Vehicles/index.md --check-prefix=MD-VEHICLES

// COM: FIXME: Add enum value comments to template

// RUN: clang-doc --format=md_mustache --doxygen --output=%t --executor=standalone %s
// RUN: FileCheck %s < %t/md/GlobalNamespace/index.md --check-prefix=MD-MUSTACHE-INDEX-LINE
// RUN: FileCheck %s < %t/md/GlobalNamespace/index.md --check-prefix=MD-MUSTACHE-INDEX
// RUN: FileCheck %s < %t/md/GlobalNamespace/_ZTV7Animals.md --check-prefix=MD-MUSTACHE-ANIMAL-LINE
// RUN: FileCheck %s < %t/md/GlobalNamespace/_ZTV7Animals.md --check-prefix=MD-MUSTACHE-ANIMAL
// RUN: FileCheck %s < %t/md/Vehicles/index.md --check-prefix=MD-MUSTACHE-VEHICLES-LINE
// RUN: FileCheck %s < %t/md/Vehicles/index.md --check-prefix=MD-MUSTACHE-VEHICLES

/**
 * @brief For specifying RGB colors
 */
enum Color {
  // MD-INDEX-LINE: *Defined at {{.*}}clang-tools-extra{{[\/]}}test{{[\/]}}clang-doc{{[\/]}}enum.cpp#[[@LINE-1]]*
  // HTML-INDEX-LINE: <p>Defined at line [[@LINE-2]] of file {{.*}}clang-tools-extra{{[\/]}}test{{[\/]}}clang-doc{{[\/]}}enum.cpp</p>
  // MD-MUSTACHE-INDEX-LINE: *Defined at {{.*}}clang-tools-extra{{[\/]}}test{{[\/]}}clang-doc{{[\/]}}enum.cpp#[[@LINE-3]]*
  Red,   ///< Comment 1
  Green, ///< Comment 2
  Blue   ///< Comment 3
};

// MD-INDEX: ## Enums
// MD-INDEX: | enum Color |
// MD-INDEX: --
// MD-INDEX: | Red |
// MD-INDEX: | Green |
// MD-INDEX: | Blue |
// MD-INDEX: **brief** For specifying RGB colors

// MD-MUSTACHE-INDEX: ## Enums
// MD-MUSTACHE-INDEX: | enum Color |
// MD-MUSTACHE-INDEX: --
// MD-MUSTACHE-INDEX: | Red |
// MD-MUSTACHE-INDEX: | Green |
// MD-MUSTACHE-INDEX: | Blue |
// MD-MUSTACHE-INDEX: **brief** For specifying RGB colors

// HTML-INDEX:     <div>
// HTML-INDEX:         <pre><code class="language-cpp code-clang-doc">enum Color</code></pre>
// HTML-INDEX:     </div>
// HTML-INDEX:     <table class="table-wrapper">
// HTML-INDEX:         <tbody>
// HTML-INDEX:             <tr>
// HTML-INDEX:                 <th>Name</th>
// HTML-INDEX:                 <th>Value</th>
// HTML-INDEX:             </tr>
// HTML-INDEX:             <tr>
// HTML-INDEX:                 <td>Red</td>
// HTML-INDEX:                 <td>0</td>
// HTML-INDEX:             </tr>
// HTML-INDEX:             <tr>
// HTML-INDEX:                 <td>Green</td>
// HTML-INDEX:                 <td>1</td>
// HTML-INDEX:             </tr>
// HTML-INDEX:             <tr>
// HTML-INDEX:                 <td>Blue</td>
// HTML-INDEX:                 <td>2</td>
// HTML-INDEX:             </tr>
// HTML-INDEX:         </tbody>
// HTML-INDEX:     </table>

/**
 * @brief Shape Types
 */
enum class Shapes {
  // MD-INDEX-LINE: *Defined at {{.*}}clang-tools-extra{{[\/]}}test{{[\/]}}clang-doc{{[\/]}}enum.cpp#[[@LINE-1]]*
  // HTML-INDEX-LINE: <p>Defined at line [[@LINE-2]] of file {{.*}}clang-tools-extra{{[\/]}}test{{[\/]}}clang-doc{{[\/]}}enum.cpp</p>
  // MD-MUSTACHE-INDEX-LINE: *Defined at {{.*}}clang-tools-extra{{[\/]}}test{{[\/]}}clang-doc{{[\/]}}enum.cpp#[[@LINE-3]]*

  /// Comment 1
  Circle,
  /// Comment 2
  Rectangle,
  /// Comment 3
  Triangle
};
// MD-INDEX: | enum class Shapes |
// MD-INDEX: --
// MD-INDEX: | Circle |
// MD-INDEX: | Rectangle |
// MD-INDEX: | Triangle |
// MD-INDEX: **brief** Shape Types

// COM: FIXME: Serialize "enum class" in template
// HTML-INDEX:     <div>
// HTML-INDEX:         <pre><code class="language-cpp code-clang-doc">enum Shapes</code></pre>
// HTML-INDEX:     </div>
// HTML-INDEX:     <table class="table-wrapper">
// HTML-INDEX:         <tbody>
// HTML-INDEX:             <tr>
// HTML-INDEX:                 <th>Name</th>
// HTML-INDEX:                 <th>Value</th>
// HTML-INDEX:             </tr>
// HTML-INDEX:             <tr>
// HTML-INDEX:                 <td>Circle</td>
// HTML-INDEX:                 <td>0</td>
// HTML-INDEX:             </tr>
// HTML-INDEX:             <tr>
// HTML-INDEX:                 <td>Rectangle</td>
// HTML-INDEX:                 <td>1</td>
// HTML-INDEX:             </tr>
// HTML-INDEX:             <tr>
// HTML-INDEX:                 <td>Triangle</td>
// HTML-INDEX:                 <td>2</td>
// HTML-INDEX:             </tr>
// HTML-INDEX:         </tbody>
// HTML-INDEX:     </table>

// COM: FIXME: Add enums declared inside of classes to class template
class Animals {
  // MD-ANIMAL-LINE: *Defined at {{.*}}clang-tools-extra{{[\/]}}test{{[\/]}}clang-doc{{[\/]}}enum.cpp#[[@LINE-1]]*
  // HTML-ANIMAL-LINE: <p>Defined at line [[@LINE-2]] of file {{.*}}clang-tools-extra{{[\/]}}test{{[\/]}}clang-doc{{[\/]}}enum.cpp</p>
  // MD-MUSTACHE-ANIMAL-LINE: *Defined at {{.*}}clang-tools-extra{{[\/]}}test{{[\/]}}clang-doc{{[\/]}}enum.cpp#[[@LINE-3]]*
public:
  /**
   * @brief specify what animal the class is
   */
  enum AnimalType {
    // MD-ANIMAL-LINE: *Defined at {{.*}}clang-tools-extra{{[\/]}}test{{[\/]}}clang-doc{{[\/]}}enum.cpp#[[@LINE-1]]*
    // HTML-ANIMAL-LINE: <p>Defined at line [[@LINE-2]] of file {{.*}}clang-tools-extra{{[\/]}}test{{[\/]}}clang-doc{{[\/]}}enum.cpp</p>
    // MD-MUSTACHE-ANIMAL-LINE: *Defined at {{.*}}clang-tools-extra{{[\/]}}test{{[\/]}}clang-doc{{[\/]}}enum.cpp#[[@LINE-3]]*
    Dog,   ///< Man's best friend
    Cat,   ///< Man's other best friend
    Iguana ///< A lizard
  };
};

// HTML-ANIMAL:      <section id="Enums" class="section-container">
// HTML-ANIMAL-NEXT:     <h2>Enumerations</h2>
// HTML-ANIMAL-NEXT:     <div id="{{([0-9A-F]{40})}}" class="delimiter-container">
// HTML-ANIMAL-NEXT:         <div>
// HTML-ANIMAL-NEXT:             <pre><code class="language-cpp code-clang-doc">enum AnimalType</code></pre>
// HTML-ANIMAL-NEXT:         </div>
// HTML-ANIMAL-NEXT:         <table class="table-wrapper">
// HTML-ANIMAL-NEXT:             <tbody>
// HTML-ANIMAL-NEXT:                 <tr>
// HTML-ANIMAL-NEXT:                     <th>Name</th>
// HTML-ANIMAL-NEXT:                     <th>Value</th>
// HTML-ANIMAL-NEXT:                 </tr>
// HTML-ANIMAL-NEXT:                 <tr>
// HTML-ANIMAL-NEXT:                     <td>Dog</td>
// HTML-ANIMAL-NEXT:                     <td>0</td>
// HTML-ANIMAL-NEXT:                 </tr>
// HTML-ANIMAL-NEXT:                 <tr>
// HTML-ANIMAL-NEXT:                     <td>Cat</td>
// HTML-ANIMAL-NEXT:                     <td>1</td>
// HTML-ANIMAL-NEXT:                 </tr>
// HTML-ANIMAL-NEXT:                 <tr>
// HTML-ANIMAL-NEXT:                     <td>Iguana</td>
// HTML-ANIMAL-NEXT:                     <td>2</td>
// HTML-ANIMAL-NEXT:                 </tr>
// HTML-ANIMAL-NEXT:             </tbody>
// HTML-ANIMAL-NEXT:         </table>
// HTML-ANIMAL-NEXT:         <div class="doc-card">
// HTML-ANIMAL-NEXT:             <div class="nested-delimiter-container">
// HTML-ANIMAL-NEXT:                 <p> specify what animal the class is</p>
// HTML-ANIMAL-NEXT:             </div>
// HTML-ANIMAL-NEXT:         </div>
// HTML-ANIMAL-NEXT:         <p>Defined at line 135 of file {{.*}}enum.cpp</p>
// HTML-ANIMAL-NEXT:     </div>
// HTML-ANIMAL-NEXT: </section>

// MD-ANIMAL: # class Animals
// MD-ANIMAL: ## Enums
// MD-ANIMAL: | enum AnimalType |
// MD-ANIMAL: --
// MD-ANIMAL: | Dog |
// MD-ANIMAL: | Cat |
// MD-ANIMAL: | Iguana |
// MD-ANIMAL: **brief** specify what animal the class is

// MD-MUSTACHE-ANIMAL: # class Animals
// MD-MUSTACHE-ANIMAL: ## Enums
// MD-MUSTACHE-ANIMAL: | enum AnimalType |
// MD-MUSTACHE-ANIMAL: --
// MD-MUSTACHE-ANIMAL: | Dog |
// MD-MUSTACHE-ANIMAL: | Cat |
// MD-MUSTACHE-ANIMAL: | Iguana |
// MD-MUSTACHE-ANIMAL: **brief** specify what animal the class is

namespace Vehicles {
/**
 * @brief specify type of car
 */
enum Car {
  // MD-VEHICLES-LINE: *Defined at {{.*}}clang-tools-extra{{[\/]}}test{{[\/]}}clang-doc{{[\/]}}enum.cpp#[[@LINE-1]]*
  // HTML-VEHICLES-LINE: Defined at line [[@LINE-2]] of file {{.*}}clang-tools-extra{{[\/]}}test{{[\/]}}clang-doc{{[\/]}}enum.cpp
  // MD-MUSTACHE-VEHICLES-LINE: *Defined at {{.*}}clang-tools-extra{{[\/]}}test{{[\/]}}clang-doc{{[\/]}}enum.cpp#[[@LINE-3]]*

  Sedan,    ///< Comment 1
  SUV,      ///< Comment 2
  Pickup,   ///< Comment 3
  Hatchback ///< Comment 4
};
} // namespace Vehicles

// MD-VEHICLES: # namespace Vehicles
// MD-VEHICLES: ## Enums
// MD-VEHICLES: | enum Car |
// MD-VEHICLES: --
// MD-VEHICLES: | Sedan |
// MD-VEHICLES: | SUV |
// MD-VEHICLES: | Pickup |
// MD-VEHICLES: | Hatchback |
// MD-VEHICLES: **brief** specify type of car

// MD-MUSTACHE-VEHICLES: # namespace Vehicles
// MD-MUSTACHE-VEHICLES: ## Enums
// MD-MUSTACHE-VEHICLES: | enum Car |
// MD-MUSTACHE-VEHICLES: --
// MD-MUSTACHE-VEHICLES: | Sedan |
// MD-MUSTACHE-VEHICLES: | SUV |
// MD-MUSTACHE-VEHICLES: | Pickup |
// MD-MUSTACHE-VEHICLES: | Hatchback |
// MD-MUSTACHE-VEHICLES: **brief** specify type of car

// HTML-VEHICLES:     <div>
// HTML-VEHICLES:         <pre><code class="language-cpp code-clang-doc">enum Car</code></pre>
// HTML-VEHICLES:      </div>
// HTML-VEHICLES:      <table class="table-wrapper">
// HTML-VEHICLES:          <tbody>
// HTML-VEHICLES:              <tr>
// HTML-VEHICLES:                  <th>Name</th>
// HTML-VEHICLES:                  <th>Value</th>
// HTML-VEHICLES:              </tr>
// HTML-VEHICLES:              <tr>
// HTML-VEHICLES:                  <td>Sedan</td>
// HTML-VEHICLES:                  <td>0</td>
// HTML-VEHICLES:              </tr>
// HTML-VEHICLES:              <tr>
// HTML-VEHICLES:                  <td>SUV</td>
// HTML-VEHICLES:                  <td>1</td>
// HTML-VEHICLES:              </tr>
// HTML-VEHICLES:              <tr>
// HTML-VEHICLES:                  <td>Pickup</td>
// HTML-VEHICLES:                  <td>2</td>
// HTML-VEHICLES:              </tr>
// HTML-VEHICLES:              <tr>
// HTML-VEHICLES:                  <td>Hatchback</td>
// HTML-VEHICLES:                  <td>3</td>
// HTML-VEHICLES:              </tr>
// HTML-VEHICLES:          </tbody>
// HTML-VEHICLES:      </table>

enum ColorUserSpecified {
  RedUserSpecified = 'A',
  GreenUserSpecified = 2,
  BlueUserSpecified = 'C'
};

// MD-INDEX: | enum ColorUserSpecified |
// MD-INDEX: --
// MD-INDEX: | RedUserSpecified |
// MD-INDEX: | GreenUserSpecified |
// MD-INDEX: | BlueUserSpecified |

// MD-MUSTACHE-INDEX: | enum ColorUserSpecified |
// MD-MUSTACHE-INDEX: --
// MD-MUSTACHE-INDEX: | RedUserSpecified |
// MD-MUSTACHE-INDEX: | GreenUserSpecified |
// MD-MUSTACHE-INDEX: | BlueUserSpecified |

// HTML-INDEX:     <div>
// HTML-INDEX:         <pre><code class="language-cpp code-clang-doc">enum ColorUserSpecified</code></pre>
// HTML-INDEX:     </div>
// HTML-INDEX:     <table class="table-wrapper">
// HTML-INDEX:         <tbody>
// HTML-INDEX:             <tr>
// HTML-INDEX:                 <th>Name</th>
// HTML-INDEX:                 <th>Value</th>
// HTML-INDEX:             </tr>
// HTML-INDEX:             <tr>
// HTML-INDEX:                 <td>RedUserSpecified</td>
// HTML-INDEX:                 <td>&#39;A&#39;</td>
// HTML-INDEX:             </tr>
// HTML-INDEX:             <tr>
// HTML-INDEX:                 <td>GreenUserSpecified</td>
// HTML-INDEX:                 <td>2</td>
// HTML-INDEX:             </tr>
// HTML-INDEX:             <tr>
// HTML-INDEX:                 <td>BlueUserSpecified</td>
// HTML-INDEX:                 <td>&#39;C&#39;</td>
// HTML-INDEX:             </tr>
// HTML-INDEX:         </tbody>
// HTML-INDEX:     </table>
