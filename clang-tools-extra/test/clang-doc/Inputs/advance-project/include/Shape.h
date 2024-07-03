#pragma once

/// Outer namespace
namespace OuterNamespace {
    /// Inner namespace
    namespace InnerNamespace {
        /**
         * @brief Enum class for colors
         */
        enum class Color {
            Red,   /**< Red color */
            Green, /**< Green color */
            Blue   /**< Blue color */
        };

        /**
         * @brief Abstract base class for shapes
         */
        class Shape {
        public:
            /**
             * @brief Constructor
             *
             * @param id The unique identifier for the shape
             */
            explicit Shape(int id);

            /**
             * @brief Virtual destructor
             */
            virtual ~Shape();

            /**
             * @brief Pure virtual function for drawing the shape
             */
            virtual void draw() const = 0;

            /**
             * @brief Getter for the shape's ID
             *
             * @return int The shape's ID
             */
            int getId() const;

        private:
            int m_id; /**< The shape's unique identifier */
        };
    }
}