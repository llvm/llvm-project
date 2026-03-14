#pragma once

/**
 * @brief Abstract base class for shapes.
 *
 * Provides a common interface for different types of shapes.
 */
class Shape {
public:
    /**
     * @brief Virtual destructor.
     */
    virtual ~Shape() {}

    /**
     * @brief Calculates the area of the shape.
     *
     * @return double The area of the shape.
     */
    virtual double area() const = 0;

    /**
     * @brief Calculates the perimeter of the shape.
     *
     * @return double The perimeter of the shape.
     */
    virtual double perimeter() const = 0;
};


