#pragma once

#include "Shape.h"
#include "Utils.h"

namespace OuterNamespace {
    namespace InnerNamespace {
        /**
         * @brief Circle class, derived from Shape
         */
        class Circle : public Shape {
        public:
            /**
             * @brief Constructor
             *
             * @param id The unique identifier for the circle
             * @param radius The radius of the circle
             */
            Circle(int id, double radius);

            /**
             * @brief Implementation of the draw function
             *
             * Draws the circle (in this case, prints circle information)
             */
            void draw() const override;

        private:
            double m_radius; /**< The radius of the circle */
        };
    }
}
