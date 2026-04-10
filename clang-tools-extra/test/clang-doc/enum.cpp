// RUN: rm -rf %t && mkdir -p %t
// RUN: clang-doc --format=html --doxygen --output=%t --executor=standalone %S/Inputs/enum.cpp
// RUN: clang-doc --format=md --doxygen --output=%t --executor=standalone %S/Inputs/enum.cpp
// RUN: clang-doc --format=md_mustache --doxygen --output=%t --executor=standalone %S/Inputs/enum.cpp
// RUN: FileCheck %s < %t/html/GlobalNamespace/index.html --check-prefix=HTML-INDEX
// RUN: FileCheck %s < %t/html/GlobalNamespace/_ZTV7Animals.html --check-prefix=HTML-ANIMAL
// RUN: FileCheck %s < %t/html/GlobalNamespace/_ZTV15FilePermissions.html --check-prefix=HTML-PERM
// RUN: FileCheck %s < %t/html/Vehicles/index.html --check-prefix=HTML-VEHICLES
// RUN: FileCheck %s < %t/GlobalNamespace/index.md --check-prefix=MD-INDEX
// RUN: FileCheck %s < %t/GlobalNamespace/Animals.md --check-prefix=MD-ANIMAL
// RUN: FileCheck %s < %t/GlobalNamespace/FilePermissions.md --check-prefix=MD-PERM
// RUN: FileCheck %s < %t/Vehicles/index.md --check-prefix=MD-VEHICLES
// RUN: FileCheck %s < %t/md/GlobalNamespace/index.md --check-prefix=MD-MUSTACHE-INDEX
// RUN: FileCheck %s < %t/md/GlobalNamespace/_ZTV7Animals.md --check-prefix=MD-MUSTACHE-ANIMAL
// RUN: FileCheck %s < %t/md/Vehicles/index.md --check-prefix=MD-MUSTACHE-VEHICLES
// RUN: FileCheck %s < %t/json/GlobalNamespace/index.json --check-prefix=JSON-INDEX
// RUN: FileCheck %s < %t/json/Vehicles/index.json --check-prefix=JSON-VEHICLES-INDEX

// JSON-INDEX:      {
// JSON-INDEX-NEXT:  "DocumentationFileName": "index",
// JSON-INDEX-NEXT:  "Enums": [
// JSON-INDEX-NEXT:    {
// JSON-INDEX-NEXT:      "Description": {
// JSON-INDEX-NEXT:        "BriefComments": [
// JSON-INDEX-NEXT:          [
// JSON-INDEX-NEXT:            {
// JSON-INDEX-NEXT:              "TextComment": "For specifying RGB colors"
// JSON-INDEX-NEXT:            }
// JSON-INDEX-NEXT:          ]
// JSON-INDEX-NEXT:        ],
// JSON-INDEX-NEXT:        "HasBriefComments": true
// JSON-INDEX-NEXT:      },
// JSON-INDEX-NEXT:      "HasComments": true,
// JSON-INDEX-NEXT:      "InfoType": "enum",
// JSON-INDEX-NEXT:      "Location": {
// JSON-INDEX-NEXT:        "Filename": "{{.*}}enum.cpp",
// JSON-INDEX-NEXT:        "LineNumber": 4
// JSON-INDEX-NEXT:      },
// JSON-INDEX-NEXT:      "Members": [
// JSON-INDEX-NEXT:        {
// JSON-INDEX-NEXT:          "Description": {
// JSON-INDEX-NEXT:            "HasParagraphComments": true,
// JSON-INDEX-NEXT:            "ParagraphComments": [
// JSON-INDEX-NEXT:              [
// JSON-INDEX-NEXT:                {
// JSON-INDEX-NEXT:                  "TextComment": "Comment 1"
// JSON-INDEX-NEXT:                }
// JSON-INDEX-NEXT:              ]
// JSON-INDEX-NEXT:            ]
// JSON-INDEX-NEXT:          },
// JSON-INDEX-NEXT:          "HasEnumMemberComments": true,
// JSON-INDEX-NEXT:          "Name": "Red",
// JSON-INDEX-NEXT:          "Value": "0"
// JSON-INDEX-NEXT:        },
// JSON-INDEX-NEXT:        {
// JSON-INDEX-NEXT:          "Description": {
// JSON-INDEX-NEXT:            "HasParagraphComments": true,
// JSON-INDEX-NEXT:            "ParagraphComments": [
// JSON-INDEX-NEXT:              [
// JSON-INDEX-NEXT:                {
// JSON-INDEX-NEXT:                  "TextComment": "Comment 2"
// JSON-INDEX-NEXT:                }
// JSON-INDEX-NEXT:              ]
// JSON-INDEX-NEXT:            ]
// JSON-INDEX-NEXT:          },
// JSON-INDEX-NEXT:          "HasEnumMemberComments": true,
// JSON-INDEX-NEXT:          "Name": "Green",
// JSON-INDEX-NEXT:          "Value": "1"
// JSON-INDEX-NEXT:        },
// JSON-INDEX-NEXT:        {
// JSON-INDEX-NEXT:          "Description": {
// JSON-INDEX-NEXT:            "HasParagraphComments": true,
// JSON-INDEX-NEXT:            "ParagraphComments": [
// JSON-INDEX-NEXT:              [
// JSON-INDEX-NEXT:                {
// JSON-INDEX-NEXT:                  "TextComment": "Comment 3"
// JSON-INDEX-NEXT:                }
// JSON-INDEX-NEXT:              ]
// JSON-INDEX-NEXT:            ]
// JSON-INDEX-NEXT:          },
// JSON-INDEX-NEXT:          "End": true,
// JSON-INDEX-NEXT:          "HasEnumMemberComments": true,
// JSON-INDEX-NEXT:          "Name": "Blue",
// JSON-INDEX-NEXT:          "Value": "2"
// JSON-INDEX-NEXT:        }
// JSON-INDEX-NEXT:      ],
// JSON-INDEX-NEXT:      "Name": "Color",
// JSON-INDEX-NEXT:      "Scoped": false,
// JSON-INDEX-NEXT:      "USR": "{{([0-9A-F]{40})}}"
// JSON-INDEX-NEXT:    },
// JSON-INDEX-NEXT:    {
// JSON-INDEX-NEXT:      "Description": {
// JSON-INDEX-NEXT:        "BriefComments": [
// JSON-INDEX-NEXT:          [
// JSON-INDEX-NEXT:            {
// JSON-INDEX-NEXT:              "TextComment": "Shape Types"
// JSON-INDEX-NEXT:            }
// JSON-INDEX-NEXT:          ]
// JSON-INDEX-NEXT:        ],
// JSON-INDEX-NEXT:        "HasBriefComments": true
// JSON-INDEX-NEXT:      },
// JSON-INDEX-NEXT:      "HasComments": true,
// JSON-INDEX-NEXT:      "InfoType": "enum",
// JSON-INDEX-NEXT:      "Location": {
// JSON-INDEX-NEXT:        "Filename": "{{.*}}enum.cpp",
// JSON-INDEX-NEXT:        "LineNumber": 13
// JSON-INDEX-NEXT:      },
// JSON-INDEX-NEXT:      "Members": [
// JSON-INDEX-NEXT:        {
// JSON-INDEX-NEXT:          "Description": {
// JSON-INDEX-NEXT:            "HasParagraphComments": true,
// JSON-INDEX-NEXT:            "ParagraphComments": [
// JSON-INDEX-NEXT:              [
// JSON-INDEX-NEXT:                {
// JSON-INDEX-NEXT:                  "TextComment": "Comment 1"
// JSON-INDEX-NEXT:                }
// JSON-INDEX-NEXT:              ]
// JSON-INDEX-NEXT:            ]
// JSON-INDEX-NEXT:          },
// JSON-INDEX-NEXT:          "HasEnumMemberComments": true,
// JSON-INDEX-NEXT:          "Name": "Circle",
// JSON-INDEX-NEXT:          "Value": "0"
// JSON-INDEX-NEXT:        },
// JSON-INDEX-NEXT:        {
// JSON-INDEX-NEXT:          "Description": {
// JSON-INDEX-NEXT:            "HasParagraphComments": true,
// JSON-INDEX-NEXT:            "ParagraphComments": [
// JSON-INDEX-NEXT:              [
// JSON-INDEX-NEXT:                {
// JSON-INDEX-NEXT:                  "TextComment": "Comment 2"
// JSON-INDEX-NEXT:                }
// JSON-INDEX-NEXT:              ]
// JSON-INDEX-NEXT:            ]
// JSON-INDEX-NEXT:          },
// JSON-INDEX-NEXT:          "HasEnumMemberComments": true,
// JSON-INDEX-NEXT:          "Name": "Rectangle",
// JSON-INDEX-NEXT:          "Value": "1"
// JSON-INDEX-NEXT:        },
// JSON-INDEX-NEXT:        {
// JSON-INDEX-NEXT:          "Description": {
// JSON-INDEX-NEXT:            "HasParagraphComments": true,
// JSON-INDEX-NEXT:            "ParagraphComments": [
// JSON-INDEX-NEXT:              [
// JSON-INDEX-NEXT:                {
// JSON-INDEX-NEXT:                  "TextComment": "Comment 3"
// JSON-INDEX-NEXT:                }
// JSON-INDEX-NEXT:              ]
// JSON-INDEX-NEXT:            ]
// JSON-INDEX-NEXT:          },
// JSON-INDEX-NEXT:          "End": true,
// JSON-INDEX-NEXT:          "HasEnumMemberComments": true,
// JSON-INDEX-NEXT:          "Name": "Triangle",
// JSON-INDEX-NEXT:          "Value": "2"
// JSON-INDEX-NEXT:        }
// JSON-INDEX-NEXT:      ],
// JSON-INDEX-NEXT:      "Name": "Shapes",
// JSON-INDEX-NEXT:      "Scoped": true,
// JSON-INDEX-NEXT:      "USR": "{{([0-9A-F]{40})}}"
// JSON-INDEX-NEXT:    },
// JSON-INDEX-NEXT:    {
// JSON-INDEX-NEXT:      "BaseType": {
// JSON-INDEX-NEXT:        "Name": "uint8_t",
// JSON-INDEX-NEXT:        "QualName": "uint8_t",
// JSON-INDEX-NEXT:        "USR": "0000000000000000000000000000000000000000"
// JSON-INDEX-NEXT:      },
// JSON-INDEX-NEXT:      "Description": {
// JSON-INDEX-NEXT:        "BriefComments": [
// JSON-INDEX-NEXT:          [
// JSON-INDEX-NEXT:            {
// JSON-INDEX-NEXT:              "TextComment": "Specify the size"
// JSON-INDEX-NEXT:            }
// JSON-INDEX-NEXT:          ]
// JSON-INDEX-NEXT:        ],
// JSON-INDEX-NEXT:        "HasBriefComments": true
// JSON-INDEX-NEXT:      },
// JSON-INDEX-NEXT:      "HasComments": true,
// JSON-INDEX-NEXT:      "InfoType": "enum",
// JSON-INDEX-NEXT:      "Location": {
// JSON-INDEX-NEXT:        "Filename": "{{.*}}enum.cpp",
// JSON-INDEX-NEXT:        "LineNumber": 26
// JSON-INDEX-NEXT:      },
// JSON-INDEX-NEXT:      "Members": [
// JSON-INDEX-NEXT:        {
// JSON-INDEX-NEXT:          "Description": {
// JSON-INDEX-NEXT:            "HasParagraphComments": true,
// JSON-INDEX-NEXT:            "ParagraphComments": [
// JSON-INDEX-NEXT:              [
// JSON-INDEX-NEXT:                {
// JSON-INDEX-NEXT:                  "TextComment": "A pearl."
// JSON-INDEX-NEXT:                },
// JSON-INDEX-NEXT:                {
// JSON-INDEX-NEXT:                  "TextComment": "Pearls are quite small."
// JSON-INDEX-NEXT:                }
// JSON-INDEX-NEXT:              ],
// JSON-INDEX-NEXT:              [
// JSON-INDEX-NEXT:                {
// JSON-INDEX-NEXT:                  "TextComment": "Pearls are used in jewelry."
// JSON-INDEX-NEXT:                }
// JSON-INDEX-NEXT:              ]
// JSON-INDEX-NEXT:            ]
// JSON-INDEX-NEXT:          },
// JSON-INDEX-NEXT:          "HasEnumMemberComments": true,
// JSON-INDEX-NEXT:          "Name": "Small",
// JSON-INDEX-NEXT:          "Value": "0"
// JSON-INDEX-NEXT:        },
// JSON-INDEX-NEXT:        {
// JSON-INDEX-NEXT:          "Description": {
// JSON-INDEX-NEXT:            "BriefComments": [
// JSON-INDEX-NEXT:              [
// JSON-INDEX-NEXT:                {
// JSON-INDEX-NEXT:                  "TextComment": "A tennis ball."
// JSON-INDEX-NEXT:                }
// JSON-INDEX-NEXT:              ]
// JSON-INDEX-NEXT:            ],
// JSON-INDEX-NEXT:            "HasBriefComments": true
// JSON-INDEX-NEXT:          },
// JSON-INDEX-NEXT:          "HasEnumMemberComments": true,
// JSON-INDEX-NEXT:          "Name": "Medium",
// JSON-INDEX-NEXT:          "Value": "1"
// JSON-INDEX-NEXT:        },
// JSON-INDEX-NEXT:        {
// JSON-INDEX-NEXT:          "Description": {
// JSON-INDEX-NEXT:            "HasParagraphComments": true,
// JSON-INDEX-NEXT:            "ParagraphComments": [
// JSON-INDEX-NEXT:              [
// JSON-INDEX-NEXT:                {
// JSON-INDEX-NEXT:                  "TextComment": "A football."
// JSON-INDEX-NEXT:                }
// JSON-INDEX-NEXT:              ]
// JSON-INDEX-NEXT:            ]
// JSON-INDEX-NEXT:          },
// JSON-INDEX-NEXT:          "End": true,
// JSON-INDEX-NEXT:          "HasEnumMemberComments": true,
// JSON-INDEX-NEXT:          "Name": "Large",
// JSON-INDEX-NEXT:          "Value": "2"
// JSON-INDEX-NEXT:        }
// JSON-INDEX-NEXT:      ],
// JSON-INDEX-NEXT:      "Name": "Size",
// JSON-INDEX-NEXT:      "Scoped": false,
// JSON-INDEX-NEXT:      "USR": "{{([0-9A-F]{40})}}"
// JSON-INDEX-NEXT:    },
// JSON-INDEX-NEXT:    {
// JSON-INDEX-NEXT:      "BaseType": {
// JSON-INDEX-NEXT:        "Name": "long long",
// JSON-INDEX-NEXT:        "QualName": "long long",
// JSON-INDEX-NEXT:        "USR": "0000000000000000000000000000000000000000"
// JSON-INDEX-NEXT:      },
// JSON-INDEX-NEXT:      "Description": {
// JSON-INDEX-NEXT:        "BriefComments": [
// JSON-INDEX-NEXT:          [
// JSON-INDEX-NEXT:            {
// JSON-INDEX-NEXT:              "TextComment": "Very long number"
// JSON-INDEX-NEXT:            }
// JSON-INDEX-NEXT:          ]
// JSON-INDEX-NEXT:        ],
// JSON-INDEX-NEXT:        "HasBriefComments": true
// JSON-INDEX-NEXT:      },
// JSON-INDEX-NEXT:      "HasComments": true,
// JSON-INDEX-NEXT:      "InfoType": "enum",
// JSON-INDEX-NEXT:      "Location": {
// JSON-INDEX-NEXT:        "Filename": "{{.*}}enum.cpp",
// JSON-INDEX-NEXT:        "LineNumber": 43
// JSON-INDEX-NEXT:      },
// JSON-INDEX-NEXT:      "Members": [
// JSON-INDEX-NEXT:        {
// JSON-INDEX-NEXT:          "Description": {
// JSON-INDEX-NEXT:            "HasParagraphComments": true,
// JSON-INDEX-NEXT:            "ParagraphComments": [
// JSON-INDEX-NEXT:              [
// JSON-INDEX-NEXT:                {
// JSON-INDEX-NEXT:                  "TextComment": "A very large value"
// JSON-INDEX-NEXT:                }
// JSON-INDEX-NEXT:              ]
// JSON-INDEX-NEXT:            ]
// JSON-INDEX-NEXT:          },
// JSON-INDEX-NEXT:          "End": true,
// JSON-INDEX-NEXT:          "HasEnumMemberComments": true,
// JSON-INDEX-NEXT:          "Name": "BigVal",
// JSON-INDEX-NEXT:          "ValueExpr": "999999999999"
// JSON-INDEX-NEXT:        }
// JSON-INDEX-NEXT:      ],
// JSON-INDEX-NEXT:      "Scoped": false,
// JSON-INDEX-NEXT:      "USR": "{{([0-9A-F]{40})}}"
// JSON-INDEX-NEXT:    },
// JSON-INDEX-NEXT:    {
// JSON-INDEX-NEXT:      "End": true,
// JSON-INDEX-NEXT:      "InfoType": "enum",
// JSON-INDEX-NEXT:      "Location": {
// JSON-INDEX-NEXT:        "Filename": "{{.*}}enum.cpp",
// JSON-INDEX-NEXT:        "LineNumber": 47
// JSON-INDEX-NEXT:      },
// JSON-INDEX-NEXT:      "Members": [
// JSON-INDEX-NEXT:        {
// JSON-INDEX-NEXT:          "Name": "RedUserSpecified",
// JSON-INDEX-NEXT:          "ValueExpr": "'A'"
// JSON-INDEX-NEXT:        },
// JSON-INDEX-NEXT:        {
// JSON-INDEX-NEXT:          "Name": "GreenUserSpecified",
// JSON-INDEX-NEXT:          "ValueExpr": "2"
// JSON-INDEX-NEXT:        },
// JSON-INDEX-NEXT:        {
// JSON-INDEX-NEXT:          "End": true,
// JSON-INDEX-NEXT:          "Name": "BlueUserSpecified",
// JSON-INDEX-NEXT:          "ValueExpr": "'C'"
// JSON-INDEX-NEXT:        }
// JSON-INDEX-NEXT:      ],
// JSON-INDEX-NEXT:      "Name": "ColorUserSpecified",
// JSON-INDEX-NEXT:      "Scoped": false,
// JSON-INDEX-NEXT:      "USR": "{{([0-9A-F]{40})}}"
// JSON-INDEX-NEXT:    }
// JSON-INDEX-NEXT:  ],

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

// MD-INDEX: ## Enums
// MD-INDEX: | enum Color |
// MD-INDEX: | Name | Value | Comments |
// MD-INDEX: |---|---|---|
// MD-INDEX: | Red | 0 | Comment 1 |
// MD-INDEX: | Green | 1 | Comment 2 |
// MD-INDEX: | Blue | 2 | Comment 3 |
// MD-INDEX: **brief** For specifying RGB colors

// MD-MUSTACHE-INDEX: ## Enums
// MD-MUSTACHE-INDEX: | enum Color |
// MD-MUSTACHE-INDEX: --
// MD-MUSTACHE-INDEX: | Red |
// MD-MUSTACHE-INDEX: | Green |
// MD-MUSTACHE-INDEX: | Blue |
// MD-MUSTACHE-INDEX: **brief** For specifying RGB colors

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

// MD-INDEX: | enum class Shapes |
// MD-INDEX: | Name | Value | Comments |
// MD-INDEX: |---|---|---|
// MD-INDEX: | Circle | 0 | Comment 1 |
// MD-INDEX: | Rectangle | 1 | Comment 2 |
// MD-INDEX: | Triangle | 2 | Comment 3 |
// MD-INDEX: **brief** Shape Types

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

// MD-INDEX: | enum Size : uint8_t |
// MD-INDEX: | Name | Value | Comments |
// MD-INDEX: |---|---|---|
// MD-INDEX: | Small | 0 | A pearl.<br>Pearls are quite small.<br><br>Pearls are used in jewelry. |
// MD-INDEX: | Medium | 1 | A tennis ball. |
// MD-INDEX: | Large | 2 | A football. |
// MD-INDEX: **brief** Specify the size

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

// MD-INDEX: | enum (unnamed) : long long |
// MD-INDEX: | Name | Value | Comments |
// MD-INDEX: |---|---|---|
// MD-INDEX: | BigVal | 999999999999 | A very large value |
// MD-INDEX: **brief** Very long number

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

// MD-INDEX: | enum ColorUserSpecified |
// MD-INDEX: | Name | Value |
// MD-INDEX: |---|---|
// MD-INDEX: | RedUserSpecified | 65 |
// MD-INDEX: | GreenUserSpecified | 2 |
// MD-INDEX: | BlueUserSpecified | 67 |

// MD-MUSTACHE-INDEX: | enum ColorUserSpecified |
// MD-MUSTACHE-INDEX: --
// MD-MUSTACHE-INDEX: | RedUserSpecified |
// MD-MUSTACHE-INDEX: | GreenUserSpecified |
// MD-MUSTACHE-INDEX: | BlueUserSpecified |

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

// MD-PERM: | enum (unnamed) |
// MD-PERM: | Name | Value | Comments |
// MD-PERM: |---|---|---|
// MD-PERM: | Read | 1 | Permission to READ r |
// MD-PERM: | Write | 2 | Permission to WRITE w |
// MD-PERM: | Execute | 4 | Permission to EXECUTE x |
// MD-PERM: **brief** File permission flags

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

// MD-ANIMAL: # class Animals
// MD-ANIMAL: ## Enums
// MD-ANIMAL: | enum AnimalType |
// MD-ANIMAL: | Name | Value | Comments |
// MD-ANIMAL: |---|---|---|
// MD-ANIMAL: | Dog | 0 | Man's best friend |
// MD-ANIMAL: | Cat | 1 | Man's other best friend |
// MD-ANIMAL: | Iguana | 2 | A lizard |
// MD-ANIMAL: **brief** specify what animal the class is

// MD-MUSTACHE-ANIMAL: # class Animals
// MD-MUSTACHE-ANIMAL: ## Enums
// MD-MUSTACHE-ANIMAL: | enum AnimalType |
// MD-MUSTACHE-ANIMAL: --
// MD-MUSTACHE-ANIMAL: | Dog |
// MD-MUSTACHE-ANIMAL: | Cat |
// MD-MUSTACHE-ANIMAL: | Iguana |
// MD-MUSTACHE-ANIMAL: **brief** specify what animal the class is

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

// MD-VEHICLES: # namespace Vehicles
// MD-VEHICLES: ## Enums
// MD-VEHICLES: | enum Car |
// MD-VEHICLES: | Name | Value | Comments |
// MD-VEHICLES: |---|---|---|
// MD-VEHICLES: | Sedan | 0 | Comment 1 |
// MD-VEHICLES: | SUV | 1 | Comment 2 |
// MD-VEHICLES: | Pickup | 2 | -- |
// MD-VEHICLES: | Hatchback | 3 | Comment 4 |
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

// JSON-VEHICLES-INDEX-LABEL:   "DocumentationFileName": "index",
// JSON-VEHICLES-INDEX-NEXT:    "Enums": [
// JSON-VEHICLES-INDEX-NEXT:      {
// JSON-VEHICLES-INDEX-NEXT:        "Description": {
// JSON-VEHICLES-INDEX-NEXT:          "BriefComments": [
// JSON-VEHICLES-INDEX-NEXT:            [
// JSON-VEHICLES-INDEX-NEXT:              {
// JSON-VEHICLES-INDEX-NEXT:                "TextComment": "specify type of car"
// JSON-VEHICLES-INDEX-NEXT:              }
// JSON-VEHICLES-INDEX-NEXT:            ]
// JSON-VEHICLES-INDEX-NEXT:          ],
// JSON-VEHICLES-INDEX-NEXT:          "HasBriefComments": true
// JSON-VEHICLES-INDEX-NEXT:        },
// JSON-VEHICLES-INDEX-NEXT:        "End": true,
// JSON-VEHICLES-INDEX-NEXT:        "HasComments": true,
// JSON-VEHICLES-INDEX-NEXT:        "InfoType": "enum",
// JSON-VEHICLES-INDEX-NEXT:        "Location": {
// JSON-VEHICLES-INDEX-NEXT:          "Filename": "{{.*}}enum.cpp",
// JSON-VEHICLES-INDEX-NEXT:          "LineNumber": 82
// JSON-VEHICLES-INDEX-NEXT:        },
// JSON-VEHICLES-INDEX-NEXT:        "Members": [
// JSON-VEHICLES-INDEX-NEXT:          {
// JSON-VEHICLES-INDEX-NEXT:            "Description": {
// JSON-VEHICLES-INDEX-NEXT:              "HasParagraphComments": true,
// JSON-VEHICLES-INDEX-NEXT:              "ParagraphComments": [
// JSON-VEHICLES-INDEX-NEXT:                [
// JSON-VEHICLES-INDEX-NEXT:                  {
// JSON-VEHICLES-INDEX-NEXT:                    "TextComment": "Comment 1"
// JSON-VEHICLES-INDEX-NEXT:                  }
// JSON-VEHICLES-INDEX-NEXT:                ]
// JSON-VEHICLES-INDEX-NEXT:              ]
// JSON-VEHICLES-INDEX-NEXT:            },
// JSON-VEHICLES-INDEX-NEXT:            "HasEnumMemberComments": true,
// JSON-VEHICLES-INDEX-NEXT:            "Name": "Sedan",
// JSON-VEHICLES-INDEX-NEXT:            "Value": "0"
// JSON-VEHICLES-INDEX-NEXT:          },
// JSON-VEHICLES-INDEX-NEXT:          {
// JSON-VEHICLES-INDEX-NEXT:            "Description": {
// JSON-VEHICLES-INDEX-NEXT:              "HasParagraphComments": true,
// JSON-VEHICLES-INDEX-NEXT:              "ParagraphComments": [
// JSON-VEHICLES-INDEX-NEXT:                [
// JSON-VEHICLES-INDEX-NEXT:                  {
// JSON-VEHICLES-INDEX-NEXT:                    "TextComment": "Comment 2"
// JSON-VEHICLES-INDEX-NEXT:                  }
// JSON-VEHICLES-INDEX-NEXT:                ]
// JSON-VEHICLES-INDEX-NEXT:              ]
// JSON-VEHICLES-INDEX-NEXT:            },
// JSON-VEHICLES-INDEX-NEXT:            "HasEnumMemberComments": true,
// JSON-VEHICLES-INDEX-NEXT:            "Name": "SUV",
// JSON-VEHICLES-INDEX-NEXT:            "Value": "1"
// JSON-VEHICLES-INDEX-NEXT:          },
// JSON-VEHICLES-INDEX-NEXT:          {
// JSON-VEHICLES-INDEX-NEXT:            "Name": "Pickup",
// JSON-VEHICLES-INDEX-NEXT:            "Value": "2"
// JSON-VEHICLES-INDEX-NEXT:          },
// JSON-VEHICLES-INDEX-NEXT:          {
// JSON-VEHICLES-INDEX-NEXT:            "Description": {
// JSON-VEHICLES-INDEX-NEXT:              "HasParagraphComments": true,
// JSON-VEHICLES-INDEX-NEXT:              "ParagraphComments": [
// JSON-VEHICLES-INDEX-NEXT:                [
// JSON-VEHICLES-INDEX-NEXT:                  {
// JSON-VEHICLES-INDEX-NEXT:                    "TextComment": "Comment 4"
// JSON-VEHICLES-INDEX-NEXT:                  }
// JSON-VEHICLES-INDEX-NEXT:                ]
// JSON-VEHICLES-INDEX-NEXT:              ]
// JSON-VEHICLES-INDEX-NEXT:            },
// JSON-VEHICLES-INDEX-NEXT:            "End": true,
// JSON-VEHICLES-INDEX-NEXT:            "HasEnumMemberComments": true,
// JSON-VEHICLES-INDEX-NEXT:            "Name": "Hatchback",
// JSON-VEHICLES-INDEX-NEXT:            "Value": "3"
// JSON-VEHICLES-INDEX-NEXT:          }
// JSON-VEHICLES-INDEX-NEXT:        ],
// JSON-VEHICLES-INDEX-NEXT:        "Name": "Car",
// JSON-VEHICLES-INDEX-NEXT:        "Namespace": [
// JSON-VEHICLES-INDEX-NEXT:          "Vehicles"
// JSON-VEHICLES-INDEX-NEXT:        ],
// JSON-VEHICLES-INDEX-NEXT:        "Scoped": false,
// JSON-VEHICLES-INDEX-NEXT:        "USR": "{{([0-9A-F]{40})}}"
// JSON-VEHICLES-INDEX-NEXT:      }
