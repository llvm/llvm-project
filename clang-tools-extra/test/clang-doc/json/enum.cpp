// RUN: rm -rf %t && mkdir -p %t
// RUN: clang-doc --output=%t --format=json --executor=standalone %s
// RUN: FileCheck %s < %t/json/GlobalNamespace/index.json --check-prefix=JSON-INDEX
// RUN: FileCheck %s < %t/json/Vehicles/index.json --check-prefix=JSON-VEHICLES-INDEX

typedef unsigned char uint8_t;
/**
 * @brief Specify the size
 */
enum Size : uint8_t {
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

// JSON-INDEX-LABEL: {
// JSON-INDEX-NEXT:    "DocumentationFileName": "index",
// JSON-INDEX-NEXT:    "Enums": [
// JSON-INDEX-NEXT:      {
// JSON-INDEX-NEXT:        "BaseType": {
// JSON-INDEX-NEXT:          "Name": "uint8_t",
// JSON-INDEX-NEXT:          "QualName": "uint8_t",
// JSON-INDEX-NEXT:          "USR": "0000000000000000000000000000000000000000"
// JSON-INDEX-NEXT:        },
// JSON-INDEX-NEXT:        "Description": {
// JSON-INDEX-NEXT:          "BriefComments": [
// JSON-INDEX-NEXT:            [
// JSON-INDEX-NEXT:              {
// JSON-INDEX-NEXT:                "TextComment": "Specify the size"
// JSON-INDEX-NEXT:              }
// JSON-INDEX-NEXT:            ]
// JSON-INDEX-NEXT:          ],
// JSON-INDEX-NEXT:          "HasBriefComments": true
// JSON-INDEX-NEXT:        },
// JSON-INDEX-NEXT:        "End": true,
// JSON-INDEX-NEXT:        "HasComments": true,
// JSON-INDEX-NEXT:        "InfoType": "enum",
// JSON-INDEX-NEXT:        "Location": {
// JSON-INDEX-NEXT:          "Filename": "{{.*}}enum.cpp",
// JSON-INDEX-NEXT:          "LineNumber": [[@LINE-38]]
// JSON-INDEX-NEXT:        },
// JSON-INDEX-NEXT:        "Members": [
// JSON-INDEX-NEXT:          {
// JSON-INDEX-NEXT:            "Description": {
// JSON-INDEX-NEXT:              "HasParagraphComments": true,
// JSON-INDEX-NEXT:              "ParagraphComments": [
// JSON-INDEX-NEXT:                [
// JSON-INDEX-NEXT:                  {
// JSON-INDEX-NEXT:                    "TextComment": "A pearl."
// JSON-INDEX-NEXT:                  },
// JSON-INDEX-NEXT:                  {
// JSON-INDEX-NEXT:                    "TextComment": "Pearls are quite small."
// JSON-INDEX-NEXT:                  }
// JSON-INDEX-NEXT:                ],
// JSON-INDEX-NEXT:                [
// JSON-INDEX-NEXT:                  {
// JSON-INDEX-NEXT:                    "TextComment": "Pearls are used in jewelry."
// JSON-INDEX-NEXT:                  }
// JSON-INDEX-NEXT:                ]
// JSON-INDEX-NEXT:              ]
// JSON-INDEX-NEXT:            },
// JSON-INDEX-NEXT:            "HasEnumMemberComments": true,
// JSON-INDEX-NEXT:            "Name": "Small",
// JSON-INDEX-NEXT:            "Value": "0"
// JSON-INDEX-NEXT:          },
// JSON-INDEX-NEXT:          {
// JSON-INDEX-NEXT:            "Description": {
// JSON-INDEX-NEXT:              "BriefComments": [
// JSON-INDEX-NEXT:                [
// JSON-INDEX-NEXT:                  {
// JSON-INDEX-NEXT:                    "TextComment": "A tennis ball."
// JSON-INDEX-NEXT:                  }
// JSON-INDEX-NEXT:                ]
// JSON-INDEX-NEXT:              ],
// JSON-INDEX-NEXT:              "HasBriefComments": true
// JSON-INDEX-NEXT:            },
// JSON-INDEX-NEXT:            "HasEnumMemberComments": true,
// JSON-INDEX-NEXT:            "Name": "Medium",
// JSON-INDEX-NEXT:            "Value": "1"
// JSON-INDEX-NEXT:          },
// JSON-INDEX-NEXT:          {
// JSON-INDEX-NEXT:            "Description": {
// JSON-INDEX-NEXT:              "HasParagraphComments": true,
// JSON-INDEX-NEXT:              "ParagraphComments": [
// JSON-INDEX-NEXT:                [
// JSON-INDEX-NEXT:                  {
// JSON-INDEX-NEXT:                    "TextComment": "A football."
// JSON-INDEX-NEXT:                  }
// JSON-INDEX-NEXT:                ]
// JSON-INDEX-NEXT:              ]
// JSON-INDEX-NEXT:            },
// JSON-INDEX-NEXT:            "End": true,
// JSON-INDEX-NEXT:            "HasEnumMemberComments": true,
// JSON-INDEX-NEXT:            "Name": "Large",
// JSON-INDEX-NEXT:            "Value": "2"
// JSON-INDEX-NEXT:          }
// JSON-INDEX-NEXT:        ],
// JSON-INDEX-NEXT:        "Name": "Size",
// JSON-INDEX-NEXT:        "Scoped": false,
// JSON-INDEX-NEXT:        "USR": "{{[0-9A-F]*}}"
// JSON-INDEX-NEXT:      }

namespace Vehicles {
/**
 * @brief specify type of car
 */
enum Car {
  Sedan,    ///< Comment 1
  SUV,      ///< Comment 2
  Pickup,
  Hatchback ///< Comment 4
};
} // namespace Vehicles

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
// JSON-VEHICLES-INDEX-NEXT:          "LineNumber": [[@LINE-26]]
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
// JSON-VEHICLES-INDEX-NEXT:        "USR": "{{[0-9A-F]*}}"
// JSON-VEHICLES-INDEX-NEXT:      }
