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
#include <cstdint>

// COM: FIXME: Add enum value comments to template

/**
 * @brief For specifying RGB colors
 */
enum Color {
  // MD-INDEX-LINE: *Defined at {{.*}}clang-tools-extra{{[\/]}}test{{[\/]}}clang-doc{{[\/]}}enum.cpp#[[@LINE-1]]*
  // HTML-INDEX-LINE: <p>Defined at line [[@LINE-2]] of file {{.*}}clang-tools-extra{{[\/]}}test{{[\/]}}clang-doc{{[\/]}}enum.cpp</p>
  Red,   ///< Comment 1
  Green, ///< Comment 2
  Blue   ///< Comment 3
};

// MD-INDEX: ## Enums
// MD-INDEX: | enum Color |
// MD-INDEX: --
// MD-INDEX: | Red | 0 |
// MD-INDEX: | Green | 1 |
// MD-INDEX: | Blue | 2 |
// MD-INDEX: **brief** For specifying RGB colors

// HTML-INDEX-LABEL:    <pre><code class="language-cpp code-clang-doc">enum Color</code></pre>
// HTML-INDEX-NEXT:     </div>
// HTML-INDEX-NEXT:     <table class="table-wrapper">
// HTML-INDEX-NEXT:         <tbody>
// HTML-INDEX-NEXT:             <tr>
// HTML-INDEX-NEXT:                 <th>Name</th>
// HTML-INDEX-NEXT:                 <th>Value</th>
// HTML-INDEX-NEXT:             </tr>
// HTML-INDEX-NEXT:             <tr>
// HTML-INDEX-NEXT:                 <td>Red</td>
// HTML-INDEX-NEXT:                 <td>0</td>
// HTML-INDEX-NEXT:             </tr>
// HTML-INDEX-NEXT:             <tr>
// HTML-INDEX-NEXT:                 <td>Green</td>
// HTML-INDEX-NEXT:                 <td>1</td>
// HTML-INDEX-NEXT:             </tr>
// HTML-INDEX-NEXT:             <tr>
// HTML-INDEX-NEXT:                 <td>Blue</td>
// HTML-INDEX-NEXT:                 <td>2</td>
// HTML-INDEX-NEXT:             </tr>
// HTML-INDEX-NEXT:         </tbody>
// HTML-INDEX-NEXT:     </table>

/**
 * @brief Shape Types
 */
enum class Shapes {
  // MD-INDEX-LINE: *Defined at {{.*}}clang-tools-extra{{[\/]}}test{{[\/]}}clang-doc{{[\/]}}enum.cpp#[[@LINE-1]]*
  // HTML-INDEX-LINE: <p>Defined at line [[@LINE-2]] of file {{.*}}clang-tools-extra{{[\/]}}test{{[\/]}}clang-doc{{[\/]}}enum.cpp</p>

  /// Comment 1
  Circle,
  /// Comment 2
  Rectangle,
  /// Comment 3
  Triangle
};
// MD-INDEX: | enum class Shapes |
// MD-INDEX: --
// MD-INDEX: | Circle | 0 |
// MD-INDEX: | Rectangle | 1 |
// MD-INDEX: | Triangle | 2 |
// MD-INDEX: **brief** Shape Types

// COM: FIXME: Serialize "enum class" in template
// HTML-INDEX-LABEL:    <pre><code class="language-cpp code-clang-doc">enum Shapes</code></pre>
// HTML-INDEX-NEXT:     </div>
// HTML-INDEX-NEXT:     <table class="table-wrapper">
// HTML-INDEX-NEXT:         <tbody>
// HTML-INDEX-NEXT:             <tr>
// HTML-INDEX-NEXT:                 <th>Name</th>
// HTML-INDEX-NEXT:                 <th>Value</th>
// HTML-INDEX-NEXT:             </tr>
// HTML-INDEX-NEXT:             <tr>
// HTML-INDEX-NEXT:                 <td>Circle</td>
// HTML-INDEX-NEXT:                 <td>0</td>
// HTML-INDEX-NEXT:             </tr>
// HTML-INDEX-NEXT:             <tr>
// HTML-INDEX-NEXT:                 <td>Rectangle</td>
// HTML-INDEX-NEXT:                 <td>1</td>
// HTML-INDEX-NEXT:             </tr>
// HTML-INDEX-NEXT:             <tr>
// HTML-INDEX-NEXT:                 <td>Triangle</td>
// HTML-INDEX-NEXT:                 <td>2</td>
// HTML-INDEX-NEXT:             </tr>
// HTML-INDEX-NEXT:         </tbody>
// HTML-INDEX-NEXT:     </table>

/**
 * @brief Specify the size
 */
enum Size : uint8_t {
  // MD-INDEX-LINE: *Defined at {{.*}}clang-tools-extra{{[\/]}}test{{[\/]}}clang-doc{{[\/]}}enum.cpp#[[@LINE-1]]*
  // HTML-INDEX-LINE: <p>Defined at line [[@LINE-2]] of file {{.*}}clang-tools-extra{{[\/]}}test{{[\/]}}clang-doc{{[\/]}}enum.cpp</p>
  Small,   ///< A pearl
  Medium,  ///< A tennis ball
  Large    ///< A football
};

// MD-INDEX: | enum Size : uint8_t |
// MD-INDEX: --
// MD-INDEX: | Small | 0 |
// MD-INDEX: | Medium | 1 |
// MD-INDEX: | Large | 2 |
// MD-INDEX: **brief** Specify the size

// HTML-INDEX:          <pre><code class="language-cpp code-clang-doc">enum Size : uint8_t</code></pre>
// HTML-INDEX-NEXT:     </div>
// HTML-INDEX-NEXT:     <table class="table-wrapper">
// HTML-INDEX-NEXT:         <tbody>
// HTML-INDEX-NEXT:             <tr>
// HTML-INDEX-NEXT:                 <th>Name</th>
// HTML-INDEX-NEXT:                 <th>Value</th>
// HTML-INDEX-NEXT:             </tr>
// HTML-INDEX-NEXT:             <tr>
// HTML-INDEX-NEXT:                 <td>Small</td>
// HTML-INDEX-NEXT:                 <td>0</td>
// HTML-INDEX-NEXT:             </tr>
// HTML-INDEX-NEXT:             <tr>
// HTML-INDEX-NEXT:                 <td>Medium</td>
// HTML-INDEX-NEXT:                 <td>1</td>
// HTML-INDEX-NEXT:             </tr>
// HTML-INDEX-NEXT:             <tr>
// HTML-INDEX-NEXT:                 <td>Large</td>
// HTML-INDEX-NEXT:                 <td>2</td>
// HTML-INDEX-NEXT:             </tr>
// HTML-INDEX-NEXT:         </tbody>
// HTML-INDEX-NEXT:     </table>

/**
 * @brief Very long number
 */
enum : long long {
  // MD-INDEX-LINE: *Defined at {{.*}}clang-tools-extra{{[\/]}}test{{[\/]}}clang-doc{{[\/]}}enum.cpp#[[@LINE-1]]*
  // HTML-INDEX-LINE: <p>Defined at line [[@LINE-2]] of file {{.*}}clang-tools-extra{{[\/]}}test{{[\/]}}clang-doc{{[\/]}}enum.cpp</p>
  BigVal = 999999999999
};

// MD-INDEX: | enum : long long |
// MD-INDEX: --
// MD-INDEX: | BigVal | 999999999999 |
// MD-INDEX: **brief** Very long number

// HTML-INDEX:          <pre><code class="language-cpp code-clang-doc">enum : long long</code></pre>
// HTML-INDEX-NEXT:     </div>
// HTML-INDEX-NEXT:     <table class="table-wrapper">
// HTML-INDEX-NEXT:         <tbody>
// HTML-INDEX-NEXT:             <tr>
// HTML-INDEX-NEXT:                 <th>Name</th>
// HTML-INDEX-NEXT:                 <th>Value</th>
// HTML-INDEX-NEXT:             </tr>
// HTML-INDEX-NEXT:             <tr>
// HTML-INDEX-NEXT:                 <td>BigVal</td>
// HTML-INDEX-NEXT:                 <td>999999999999</td>
// HTML-INDEX-NEXT:             </tr>
// HTML-INDEX-NEXT:         </tbody>
// HTML-INDEX-NEXT:     </table>

/**
 * @brief This is a top secret
 */
enum {
  // MD-INDEX-LINE: *Defined at {{.*}}clang-tools-extra{{[\/]}}test{{[\/]}}clang-doc{{[\/]}}enum.cpp#[[@LINE-1]]*
  // HTML-INDEX-LINE: <p>Defined at line [[@LINE-2]] of file {{.*}}clang-tools-extra{{[\/]}}test{{[\/]}}clang-doc{{[\/]}}enum.cpp</p>
  secret = true
};

// MD-INDEX: | enum |
// MD-INDEX: --
// MD-INDEX: | secret | 1 |
// MD-INDEX: **brief** This is a top secret

// HTML-INDEX:          <pre><code class="language-cpp code-clang-doc">enum </code></pre>
// HTML-INDEX-NEXT:     </div>
// HTML-INDEX-NEXT:     <table class="table-wrapper">
// HTML-INDEX-NEXT:         <tbody>
// HTML-INDEX-NEXT:             <tr>
// HTML-INDEX-NEXT:                 <th>Name</th>
// HTML-INDEX-NEXT:                 <th>Value</th>
// HTML-INDEX-NEXT:             </tr>
// HTML-INDEX-NEXT:             <tr>
// HTML-INDEX-NEXT:                 <td>secret</td>
// HTML-INDEX-NEXT:                 <td>true</td>
// HTML-INDEX-NEXT:             </tr>
// HTML-INDEX-NEXT:         </tbody>
// HTML-INDEX-NEXT:     </table>

// COM: FIXME: Add enums declared inside of classes to class template
class Animals {
  // MD-ANIMAL-LINE: *Defined at {{.*}}clang-tools-extra{{[\/]}}test{{[\/]}}clang-doc{{[\/]}}enum.cpp#[[@LINE-1]]*
  // HTML-ANIMAL-LINE: <p>Defined at line [[@LINE-2]] of file {{.*}}clang-tools-extra{{[\/]}}test{{[\/]}}clang-doc{{[\/]}}enum.cpp</p>
public:
  /**
   * @brief specify what animal the class is
   */
  enum AnimalType {
    // MD-ANIMAL-LINE: *Defined at {{.*}}clang-tools-extra{{[\/]}}test{{[\/]}}clang-doc{{[\/]}}enum.cpp#[[@LINE-1]]*
    // HTML-ANIMAL-LINE: <p>Defined at line [[@LINE-2]] of file {{.*}}clang-tools-extra{{[\/]}}test{{[\/]}}clang-doc{{[\/]}}enum.cpp</p>
    Dog,   ///< Man's best friend
    Cat,   ///< Man's other best friend
    Iguana ///< A lizard
  };
};

// HTML-ANIMAL:          <section id="Enums" class="section-container">
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
// HTML-ANIMAL-NEXT:         <p>Defined at line {{.*}} of file {{.*}}enum.cpp</p>
// HTML-ANIMAL-NEXT:     </div>
// HTML-ANIMAL-NEXT: </section>

// MD-ANIMAL: # class Animals
// MD-ANIMAL: ## Enums
// MD-ANIMAL: | enum AnimalType |
// MD-ANIMAL: --
// MD-ANIMAL: | Dog | 0 |
// MD-ANIMAL: | Cat | 1 |
// MD-ANIMAL: | Iguana | 2 |
// MD-ANIMAL: **brief** specify what animal the class is

namespace Vehicles {
/**
 * @brief specify type of car
 */
enum Car {
  // MD-VEHICLES-LINE: *Defined at {{.*}}clang-tools-extra{{[\/]}}test{{[\/]}}clang-doc{{[\/]}}enum.cpp#[[@LINE-1]]*
  // HTML-VEHICLES-LINE: Defined at line [[@LINE-2]] of file {{.*}}clang-tools-extra{{[\/]}}test{{[\/]}}clang-doc{{[\/]}}enum.cpp

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
// MD-VEHICLES: | Sedan | 0 |
// MD-VEHICLES: | SUV | 1 |
// MD-VEHICLES: | Pickup | 2 |
// MD-VEHICLES: | Hatchback | 3 |
// MD-VEHICLES: **brief** specify type of car

// HTML-VEHICLES-LABEL:     <pre><code class="language-cpp code-clang-doc">enum Car</code></pre>
// HTML-VEHICLES-NEXT:      </div>
// HTML-VEHICLES-NEXT:      <table class="table-wrapper">
// HTML-VEHICLES-NEXT:          <tbody>
// HTML-VEHICLES-NEXT:              <tr>
// HTML-VEHICLES-NEXT:                  <th>Name</th>
// HTML-VEHICLES-NEXT:                  <th>Value</th>
// HTML-VEHICLES-NEXT:              </tr>
// HTML-VEHICLES-NEXT:              <tr>
// HTML-VEHICLES-NEXT:                  <td>Sedan</td>
// HTML-VEHICLES-NEXT:                  <td>0</td>
// HTML-VEHICLES-NEXT:              </tr>
// HTML-VEHICLES-NEXT:              <tr>
// HTML-VEHICLES-NEXT:                  <td>SUV</td>
// HTML-VEHICLES-NEXT:                  <td>1</td>
// HTML-VEHICLES-NEXT:              </tr>
// HTML-VEHICLES-NEXT:              <tr>
// HTML-VEHICLES-NEXT:                  <td>Pickup</td>
// HTML-VEHICLES-NEXT:                  <td>2</td>
// HTML-VEHICLES-NEXT:              </tr>
// HTML-VEHICLES-NEXT:              <tr>
// HTML-VEHICLES-NEXT:                  <td>Hatchback</td>
// HTML-VEHICLES-NEXT:                  <td>3</td>
// HTML-VEHICLES-NEXT:              </tr>
// HTML-VEHICLES-NEXT:          </tbody>
// HTML-VEHICLES-NEXT:      </table>

enum ColorUserSpecified {
  RedUserSpecified = 'A',
  GreenUserSpecified = 2,
  BlueUserSpecified = 'C'
};

// MD-INDEX: | enum ColorUserSpecified |
// MD-INDEX: --
// MD-INDEX: | RedUserSpecified | 65 |
// MD-INDEX: | GreenUserSpecified | 2 |
// MD-INDEX: | BlueUserSpecified | 67 |

// HTML-INDEX-LABEL:    <pre><code class="language-cpp code-clang-doc">enum ColorUserSpecified</code></pre>
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
