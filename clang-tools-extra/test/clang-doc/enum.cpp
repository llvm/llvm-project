// RUN: rm -rf %t && mkdir -p %t
// RUN: clang-doc --format=html --doxygen --output=%t --executor=standalone %s
// RUN: clang-doc --format=md --doxygen --output=%t --executor=standalone %s
// RUN: clang-doc --format=mustache --doxygen --output=%t --executor=standalone %s
// RUN: FileCheck %s < %t/GlobalNamespace/index.html --check-prefix=HTML-INDEX-LINE
// RUN: FileCheck %s < %t/GlobalNamespace/index.html --check-prefix=HTML-INDEX
// RUN: FileCheck %s < %t/GlobalNamespace/Animals.html --check-prefix=HTML-ANIMAL-LINE
// RUN: FileCheck %s < %t/GlobalNamespace/Animals.html --check-prefix=HTML-ANIMAL
// RUN: FileCheck %s < %t/Vehicles/index.html --check-prefix=HTML-VEHICLES-LINE
// RUN: FileCheck %s < %t/Vehicles/index.html --check-prefix=HTML-VEHICLES
// RUN: FileCheck %s < %t/html/GlobalNamespace/index.html --check-prefix=MUSTACHE-INDEX-LINE
// RUN: FileCheck %s < %t/html/GlobalNamespace/index.html --check-prefix=MUSTACHE-INDEX
// RUN: FileCheck %s < %t/html/GlobalNamespace/_ZTV7Animals.html --check-prefix=MUSTACHE-ANIMAL-LINE
// RUN: FileCheck %s < %t/html/GlobalNamespace/_ZTV7Animals.html --check-prefix=MUSTACHE-ANIMAL
// RUN: FileCheck %s < %t/html/Vehicles/index.html --check-prefix=MUSTACHE-VEHICLES-LINE
// RUN: FileCheck %s < %t/html/Vehicles/index.html --check-prefix=MUSTACHE-VEHICLES
// RUN: FileCheck %s < %t/GlobalNamespace/index.md --check-prefix=MD-INDEX-LINE
// RUN: FileCheck %s < %t/GlobalNamespace/index.md --check-prefix=MD-INDEX
// RUN: FileCheck %s < %t/GlobalNamespace/Animals.md --check-prefix=MD-ANIMAL-LINE
// RUN: FileCheck %s < %t/GlobalNamespace/Animals.md --check-prefix=MD-ANIMAL
// RUN: FileCheck %s < %t/Vehicles/index.md --check-prefix=MD-VEHICLES-LINE
// RUN: FileCheck %s < %t/Vehicles/index.md --check-prefix=MD-VEHICLES

// COM: FIXME: Add enum value comments to template

/**
 * @brief For specifying RGB colors
 */
enum Color {
  // MD-INDEX-LINE: *Defined at {{.*}}clang-tools-extra{{[\/]}}test{{[\/]}}clang-doc{{[\/]}}enum.cpp#[[@LINE-1]]*
  // HTML-INDEX-LINE: <p>Defined at line [[@LINE-2]] of file {{.*}}clang-tools-extra{{[\/]}}test{{[\/]}}clang-doc{{[\/]}}enum.cpp</p>
  // MUSTACHE-INDEX-LINE-NOT: <p>Defined at line [[@LINE-2]] of file {{.*}}clang-tools-extra{{[\/]}}test{{[\/]}}clang-doc{{[\/]}}enum.cpp</p>
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

// HTML-INDEX: <th colspan="3">enum Color</th>
// HTML-INDEX: <td>Red</td>
// HTML-INDEX: <td>0</td>
// HTML-INDEX: <p> Comment 1</p>
// HTML-INDEX: <td>Green</td>
// HTML-INDEX: <td>1</td>
// HTML-INDEX: <p> Comment 2</p>
// HTML-INDEX: <td>Blue</td>
// HTML-INDEX: <td>2</td>
// HTML-INDEX: <p> Comment 3</p>

// MUSTACHE-INDEX:     <div>
// MUSTACHE-INDEX:         <pre>
// MUSTACHE-INDEX:             <code class="language-cpp code-clang-doc">
// MUSTACHE-INDEX: enum Color
// MUSTACHE-INDEX:             </code>
// MUSTACHE-INDEX:         </pre>
// MUSTACHE-INDEX:     </div>
// MUSTACHE-INDEX:     <table class="table-wrapper">
// MUSTACHE-INDEX:         <tbody>
// MUSTACHE-INDEX:             <tr>
// MUSTACHE-INDEX:                 <th>Name</th>
// MUSTACHE-INDEX:                 <th>Value</th>
// MUSTACHE-INDEX:             </tr>
// MUSTACHE-INDEX:                 <tr>
// MUSTACHE-INDEX:                     <td>Red</td>
// MUSTACHE-INDEX:                     <td>0</td>
// MUSTACHE-INDEX:                 </tr>
// MUSTACHE-INDEX:                 <tr>
// MUSTACHE-INDEX:                     <td>Green</td>
// MUSTACHE-INDEX:                     <td>1</td>
// MUSTACHE-INDEX:                 </tr>
// MUSTACHE-INDEX:                 <tr>
// MUSTACHE-INDEX:                     <td>Blue</td>
// MUSTACHE-INDEX:                     <td>2</td>
// MUSTACHE-INDEX:                 </tr>
// MUSTACHE-INDEX:         </tbody>
// MUSTACHE-INDEX:     </table>

/**
 * @brief Shape Types
 */
enum class Shapes {
  // MD-INDEX-LINE: *Defined at {{.*}}clang-tools-extra{{[\/]}}test{{[\/]}}clang-doc{{[\/]}}enum.cpp#[[@LINE-1]]*
  // HTML-INDEX-LINE: <p>Defined at line [[@LINE-2]] of file {{.*}}clang-tools-extra{{[\/]}}test{{[\/]}}clang-doc{{[\/]}}enum.cpp</p>
  // MUSTACHE-INDEX-LINE-NOT: <p>Defined at line [[@LINE-2]] of file {{.*}}clang-tools-extra{{[\/]}}test{{[\/]}}clang-doc{{[\/]}}enum.cpp</p>

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

// HTML-INDEX: <th colspan="3">enum class Shapes</th>
// HTML-INDEX: <td>Circle</td>
// HTML-INDEX: <td>0</td>
// HTML-INDEX: <p> Comment 1</p>
// HTML-INDEX: <td>Rectangle</td>
// HTML-INDEX: <td>1</td>
// HTML-INDEX: <p> Comment 2</p>
// HTML-INDEX: <td>Triangle</td>
// HTML-INDEX: <td>2</td>
// HTML-INDEX: <p> Comment 3</p>

// COM: FIXME: Serialize "enum class" in template
// MUSTACHE-INDEX:     <div>
// MUSTACHE-INDEX:         <pre>
// MUSTACHE-INDEX:             <code class="language-cpp code-clang-doc">
// MUSTACHE-INDEX: enum Shapes
// MUSTACHE-INDEX:             </code>
// MUSTACHE-INDEX:         </pre>
// MUSTACHE-INDEX:     </div>
// MUSTACHE-INDEX:     <table class="table-wrapper">
// MUSTACHE-INDEX:         <tbody>
// MUSTACHE-INDEX:             <tr>
// MUSTACHE-INDEX:                 <th>Name</th>
// MUSTACHE-INDEX:                 <th>Value</th>
// MUSTACHE-INDEX:             </tr>
// MUSTACHE-INDEX:                 <tr>
// MUSTACHE-INDEX:                     <td>Circle</td>
// MUSTACHE-INDEX:                     <td>0</td>
// MUSTACHE-INDEX:                 </tr>
// MUSTACHE-INDEX:                 <tr>
// MUSTACHE-INDEX:                     <td>Rectangle</td>
// MUSTACHE-INDEX:                     <td>1</td>
// MUSTACHE-INDEX:                 </tr>
// MUSTACHE-INDEX:                 <tr>
// MUSTACHE-INDEX:                     <td>Triangle</td>
// MUSTACHE-INDEX:                     <td>2</td>
// MUSTACHE-INDEX:                 </tr>
// MUSTACHE-INDEX:         </tbody>
// MUSTACHE-INDEX:     </table>

// COM: FIXME: Add enums declared inside of classes to class template
class Animals {
  // MD-ANIMAL-LINE: *Defined at {{.*}}clang-tools-extra{{[\/]}}test{{[\/]}}clang-doc{{[\/]}}enum.cpp#[[@LINE-1]]*
  // HTML-ANIMAL-LINE: <p>Defined at line [[@LINE-2]] of file {{.*}}clang-tools-extra{{[\/]}}test{{[\/]}}clang-doc{{[\/]}}enum.cpp</p>
  // MUSTACHE-ANIMAL-LINE: <p>Defined at line [[@LINE-3]] of file {{.*}}clang-tools-extra{{[\/]}}test{{[\/]}}clang-doc{{[\/]}}enum.cpp</p>
public:
  /**
   * @brief specify what animal the class is
   */
  enum AnimalType {
    // MD-ANIMAL-LINE: *Defined at {{.*}}clang-tools-extra{{[\/]}}test{{[\/]}}clang-doc{{[\/]}}enum.cpp#[[@LINE-1]]*
    // HTML-ANIMAL-LINE: <p>Defined at line [[@LINE-2]] of file {{.*}}clang-tools-extra{{[\/]}}test{{[\/]}}clang-doc{{[\/]}}enum.cpp</p>
    // MUSTACHE-ANIMAL-LINE-NOT: <p>Defined at line [[@LINE-2]] of file {{.*}}clang-tools-extra{{[\/]}}test{{[\/]}}clang-doc{{[\/]}}enum.cpp</p>
    Dog,   ///< Man's best friend
    Cat,   ///< Man's other best friend
    Iguana ///< A lizard
  };
};

// HTML-ANIMAL: <h1>class Animals</h1>
// HTML-ANIMAL: <h2 id="Enums">Enums</h2>
// HTML-ANIMAL: <th colspan="3">enum AnimalType</th>
// HTML-ANIMAL: <td>Dog</td>
// HTML-ANIMAL: <td>0</td>
// HTML-ANIMAL: <p> Man&apos;s best friend</p>
// HTML-ANIMAL: <td>Cat</td>
// HTML-ANIMAL: <td>1</td>
// HTML-ANIMAL: <p> Man&apos;s other best friend</p>
// HTML-ANIMAL: <td>Iguana</td>
// HTML-ANIMAL: <td>2</td>
// HTML-ANIMAL: <p> A lizard</p>

// MUSTACHE-ANIMAL-NOT: <h1>class Animals</h1>
// MUSTACHE-ANIMAL-NOT: <h2 id="Enums">Enums</h2>
// MUSTACHE-ANIMAL-NOT: <th colspan="3">enum AnimalType</th>
// MUSTACHE-ANIMAL-NOT: <td>Dog</td>
// MUSTACHE-ANIMAL-NOT: <td>0</td>
// MUSTACHE-ANIMAL-NOT: <p> Man&apos;s best friend</p>
// MUSTACHE-ANIMAL-NOT: <td>Cat</td>
// MUSTACHE-ANIMAL-NOT: <td>1</td>
// MUSTACHE-ANIMAL-NOT: <p> Man&apos;s other best friend</p>
// MUSTACHE-ANIMAL-NOT: <td>Iguana</td>
// MUSTACHE-ANIMAL-NOT: <td>2</td>
// MUSTACHE-ANIMAL-NOT: <p> A lizard</p>

// MD-ANIMAL: # class Animals
// MD-ANIMAL: ## Enums
// MD-ANIMAL: | enum AnimalType |
// MD-ANIMAL: --
// MD-ANIMAL: | Dog |
// MD-ANIMAL: | Cat |
// MD-ANIMAL: | Iguana |
// MD-ANIMAL: **brief** specify what animal the class is

namespace Vehicles {
/**
 * @brief specify type of car
 */
enum Car {
  // MD-VEHICLES-LINE: *Defined at {{.*}}clang-tools-extra{{[\/]}}test{{[\/]}}clang-doc{{[\/]}}enum.cpp#[[@LINE-1]]*
  // HTML-VEHICLES-LINE: <p>Defined at line [[@LINE-2]] of file {{.*}}clang-tools-extra{{[\/]}}test{{[\/]}}clang-doc{{[\/]}}enum.cpp</p>
  // MUSTACHE-VEHICLES-LINE: Defined at line [[@LINE-3]] of file {{.*}}clang-tools-extra{{[\/]}}test{{[\/]}}clang-doc{{[\/]}}enum.cpp

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

// HTML-VEHICLES: <h1>namespace Vehicles</h1>
// HTML-VEHICLES: <th colspan="3">enum Car</th>
// HTML-VEHICLES: <td>Sedan</td>
// HTML-VEHICLES: <td>0</td>
// HTML-VEHICLES: <p> Comment 1</p>
// HTML-VEHICLES: <td>SUV</td>
// HTML-VEHICLES: <td>1</td>
// HTML-VEHICLES: <p> Comment 2</p>
// HTML-VEHICLES: <td>Pickup</td>
// HTML-VEHICLES: <td>2</td>
// HTML-VEHICLES: <p> Comment 3</p>
// HTML-VEHICLES: <td>Hatchback</td>
// HTML-VEHICLES: <td>3</td>
// HTML-VEHICLES: <p> Comment 4</p>

// MUSTACHE-VEHICLES:     <div>
// MUSTACHE-VEHICLES:         <pre>
// MUSTACHE-VEHICLES:             <code class="language-cpp code-clang-doc">
// MUSTACHE-VEHICLES: enum Car
// MUSTACHE-VEHICLES:             </code>
// MUSTACHE-VEHICLES:         </pre>
// MUSTACHE-VEHICLES:      </div>
// MUSTACHE-VEHICLES:      <table class="table-wrapper">
// MUSTACHE-VEHICLES:          <tbody>
// MUSTACHE-VEHICLES:              <tr>
// MUSTACHE-VEHICLES:                  <th>Name</th>
// MUSTACHE-VEHICLES:                  <th>Value</th>
// MUSTACHE-VEHICLES:              </tr>
// MUSTACHE-VEHICLES:                  <tr>
// MUSTACHE-VEHICLES:                      <td>Sedan</td>
// MUSTACHE-VEHICLES:                      <td>0</td>
// MUSTACHE-VEHICLES:                  </tr>
// MUSTACHE-VEHICLES:                  <tr>
// MUSTACHE-VEHICLES:                      <td>SUV</td>
// MUSTACHE-VEHICLES:                      <td>1</td>
// MUSTACHE-VEHICLES:                  </tr>
// MUSTACHE-VEHICLES:                  <tr>
// MUSTACHE-VEHICLES:                      <td>Pickup</td>
// MUSTACHE-VEHICLES:                      <td>2</td>
// MUSTACHE-VEHICLES:                  </tr>
// MUSTACHE-VEHICLES:                  <tr>
// MUSTACHE-VEHICLES:                      <td>Hatchback</td>
// MUSTACHE-VEHICLES:                      <td>3</td>
// MUSTACHE-VEHICLES:                  </tr>
// MUSTACHE-VEHICLES:          </tbody>
// MUSTACHE-VEHICLES:      </table>

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

// HTML-INDEX: <th colspan="2">enum ColorUserSpecified</th>
// HTML-INDEX: <td>RedUserSpecified</td>
// HTML-INDEX: <td>&apos;A&apos;</td>
// HTML-INDEX: <td>GreenUserSpecified</td>
// HTML-INDEX: <td>2</td>
// HTML-INDEX: <td>BlueUserSpecified</td>
// HTML-INDEX: <td>&apos;C&apos;</td>

// MUSTACHE-INDEX:     <div>
// MUSTACHE-INDEX:         <pre>
// MUSTACHE-INDEX:             <code class="language-cpp code-clang-doc">
// MUSTACHE-INDEX: enum ColorUserSpecified
// MUSTACHE-INDEX:             </code>
// MUSTACHE-INDEX:         </pre>
// MUSTACHE-INDEX:     </div>
// MUSTACHE-INDEX:     <table class="table-wrapper">
// MUSTACHE-INDEX:         <tbody>
// MUSTACHE-INDEX:             <tr>
// MUSTACHE-INDEX:                 <th>Name</th>
// MUSTACHE-INDEX:                 <th>Value</th>
// MUSTACHE-INDEX:             </tr>
// MUSTACHE-INDEX:                 <tr>
// MUSTACHE-INDEX:                     <td>RedUserSpecified</td>
// MUSTACHE-INDEX:                     <td>&#39;A&#39;</td>
// MUSTACHE-INDEX:                 </tr>
// MUSTACHE-INDEX:                 <tr>
// MUSTACHE-INDEX:                     <td>GreenUserSpecified</td>
// MUSTACHE-INDEX:                     <td>2</td>
// MUSTACHE-INDEX:                 </tr>
// MUSTACHE-INDEX:                 <tr>
// MUSTACHE-INDEX:                     <td>BlueUserSpecified</td>
// MUSTACHE-INDEX:                     <td>&#39;C&#39;</td>
// MUSTACHE-INDEX:                 </tr>
// MUSTACHE-INDEX:         </tbody>
// MUSTACHE-INDEX:     </table>
