// RUN: rm -rf %t && mkdir -p %t
// RUN: clang-doc --format=html --doxygen --output=%t --executor=standalone %s
// RUN: clang-doc --format=md --doxygen --output=%t --executor=standalone %s
// RUN: FileCheck %s < %t/html/GlobalNamespace/index.html --check-prefix=HTML-INDEX-LINE
// RUN: FileCheck %s < %t/html/GlobalNamespace/index.html --check-prefix=HTML-INDEX
// RUN: FileCheck %s < %t/html/GlobalNamespace/_ZTV7Animals.html --check-prefix=HTML-ANIMAL-LINE
// RUN: FileCheck %s < %t/html/GlobalNamespace/_ZTV7Animals.html --check-prefix=HTML-ANIMAL
// RUN: FileCheck %s < %t/html/GlobalNamespace/_ZTV15FilePermissions.html --check-prefix=HTML-PERM-LINE
// RUN: FileCheck %s < %t/html/GlobalNamespace/_ZTV15FilePermissions.html --check-prefix=HTML-PERM
// RUN: FileCheck %s < %t/html/Vehicles/index.html --check-prefix=HTML-VEHICLES-LINE
// RUN: FileCheck %s < %t/html/Vehicles/index.html --check-prefix=HTML-VEHICLES
// RUN: FileCheck %s < %t/GlobalNamespace/index.md --check-prefix=MD-INDEX-LINE
// RUN: FileCheck %s < %t/GlobalNamespace/index.md --check-prefix=MD-INDEX
// RUN: FileCheck %s < %t/GlobalNamespace/Animals.md --check-prefix=MD-ANIMAL-LINE
// RUN: FileCheck %s < %t/GlobalNamespace/Animals.md --check-prefix=MD-ANIMAL
// RUN: FileCheck %s < %t/GlobalNamespace/FilePermissions.md --check-prefix=MD-PERM-LINE
// RUN: FileCheck %s < %t/GlobalNamespace/FilePermissions.md --check-prefix=MD-PERM
// RUN: FileCheck %s < %t/Vehicles/index.md --check-prefix=MD-VEHICLES-LINE
// RUN: FileCheck %s < %t/Vehicles/index.md --check-prefix=MD-VEHICLES


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
// HTML-INDEX-NEXT:     <p>Defined at line [[@LINE-62]] of file {{.*}}clang-tools-extra{{[\/]}}test{{[\/]}}clang-doc{{[\/]}}enum.cpp</p>
// HTML-INDEX-NEXT:   </div>

// MD-MUSTACHE-INDEX: ## Enums
// MD-MUSTACHE-INDEX: | enum Color |
// MD-MUSTACHE-INDEX: --
// MD-MUSTACHE-INDEX: | Red |
// MD-MUSTACHE-INDEX: | Green |
// MD-MUSTACHE-INDEX: | Blue |
// MD-MUSTACHE-INDEX: **brief** For specifying RGB colors

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
// HTML-INDEX-NEXT:     <p>Defined at line [[@LINE-64]] of file {{.*}}clang-tools-extra{{[\/]}}test{{[\/]}}clang-doc{{[\/]}}enum.cpp</p>
// HTML-INDEX-NEXT:   </div>

typedef unsigned char uint8_t;
/**
 * @brief Specify the size
 */
enum Size : uint8_t {
  // MD-INDEX-LINE: *Defined at {{.*}}clang-tools-extra{{[\/]}}test{{[\/]}}clang-doc{{[\/]}}enum.cpp#[[@LINE-1]]*
  // HTML-INDEX-LINE: <p>Defined at line [[@LINE-2]] of file {{.*}}clang-tools-extra{{[\/]}}test{{[\/]}}clang-doc{{[\/]}}enum.cpp</p>

  /// A pearl.
  /// Pearls are quite small.
  ///
  /// Pearls are used in jewelry.
  Small,

  /// @brief A tennis ball.
  Medium,

  /// A football.
  Large
};

// MD-INDEX: | enum Size : uint8_t |
// MD-INDEX: --
// MD-INDEX: | Small |
// MD-INDEX: | Medium |
// MD-INDEX: | Large |
// MD-INDEX: **brief** Specify the size

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
// HTML-INDEX-NEXT:     <p>Defined at line [[@LINE-71]] of file {{.*}}clang-tools-extra{{[\/]}}test{{[\/]}}clang-doc{{[\/]}}enum.cpp</p>
// HTML-INDEX-NEXT:   </div>

/**
 * @brief Very long number
 */
enum : long long {
  // MD-INDEX-LINE: *Defined at {{.*}}clang-tools-extra{{[\/]}}test{{[\/]}}clang-doc{{[\/]}}enum.cpp#[[@LINE-1]]*
  // HTML-INDEX-LINE: <p>Defined at line [[@LINE-2]] of file {{.*}}clang-tools-extra{{[\/]}}test{{[\/]}}clang-doc{{[\/]}}enum.cpp</p>
  BigVal = 999999999999   ///< A very large value
};

// MD-INDEX: | enum (unnamed) : long long |
// MD-INDEX: --
// MD-INDEX: | BigVal |
// MD-INDEX: **brief** Very long number

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
// HTML-INDEX-NEXT:     <p>Defined at line [[@LINE-38]] of file {{.*}}clang-tools-extra{{[\/]}}test{{[\/]}}clang-doc{{[\/]}}enum.cpp</p>
// HTML-INDEX-NEXT:   </div>

class FilePermissions {
// MD-PERM-LINE: *Defined at {{.*}}clang-tools-extra{{[\/]}}test{{[\/]}}clang-doc{{[\/]}}enum.cpp#[[@LINE-1]]*
// HTML-PERM-LINE: <p>Defined at line [[@LINE-2]] of file {{.*}}clang-tools-extra{{[\/]}}test{{[\/]}}clang-doc{{[\/]}}enum.cpp</p>
public:
  /**
   * @brief File permission flags
   */
  enum {
  // MD-PERM-LINE: *Defined at {{.*}}clang-tools-extra{{[\/]}}test{{[\/]}}clang-doc{{[\/]}}enum.cpp#[[@LINE-1]]*
  // HTML-PERM-LINE: <p>Defined at line [[@LINE-2]] of file {{.*}}clang-tools-extra{{[\/]}}test{{[\/]}}clang-doc{{[\/]}}enum.cpp</p>
    Read    = 1,     ///< Permission to READ r
    Write   = 2,     ///< Permission to WRITE w
    Execute = 4      ///< Permission to EXECUTE x
  };
};

// MD-PERM: | enum (unnamed) |
// MD-PERM: --
// MD-PERM: | Read |
// MD-PERM: | Write |
// MD-PERM: | Execute |
// MD-PERM: **brief** File permission flags

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
// HTML-PERM-NEXT:       </div>
// HTML-PERM-NEXT:         <p>Defined at line [[@LINE-63]] of file {{.*}}clang-tools-extra{{[\/]}}test{{[\/]}}clang-doc{{[\/]}}enum.cpp</p>
// HTML-PERM-NEXT:     </div>
// HTML-PERM-NEXT:   </section>

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
// HTML-ANIMAL-NEXT:         <p>Defined at line [[@LINE-57]] of file {{.*}}enum.cpp</p>
// HTML-ANIMAL-NEXT:      </div>
// HTML-ANIMAL-NEXT:    </section>

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
  Pickup,
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
// HTML-VEHICLES-NEXT:      <p>Defined at line [[@LINE-72]] of file {{.*}}clang-tools-extra{{[\/]}}test{{[\/]}}clang-doc{{[\/]}}enum.cpp</p>
// HTML-VEHICLES-NEXT:    </div>

// MD-MUSTACHE-VEHICLES: # namespace Vehicles
// MD-MUSTACHE-VEHICLES: ## Enums
// MD-MUSTACHE-VEHICLES: | enum Car |
// MD-MUSTACHE-VEHICLES: --
// MD-MUSTACHE-VEHICLES: | Sedan |
// MD-MUSTACHE-VEHICLES: | SUV |
// MD-MUSTACHE-VEHICLES: | Pickup |
// MD-MUSTACHE-VEHICLES: | Hatchback |
// MD-MUSTACHE-VEHICLES: **brief** specify type of car

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
// HTML-INDEX-NEXT:     <p>Defined at line [[@LINE-36]] of file {{.*}}clang-tools-extra{{[\/]}}test{{[\/]}}clang-doc{{[\/]}}enum.cpp</p>
// HTML-INDEX-NEXT:   </div>

// MD-MUSTACHE-INDEX: | enum ColorUserSpecified |
// MD-MUSTACHE-INDEX: --
// MD-MUSTACHE-INDEX: | RedUserSpecified |
// MD-MUSTACHE-INDEX: | GreenUserSpecified |
// MD-MUSTACHE-INDEX: | BlueUserSpecified |
