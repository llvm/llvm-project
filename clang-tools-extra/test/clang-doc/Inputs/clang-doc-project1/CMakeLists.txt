project(clang_doc_project1)

set(CMAKE_CXX_STANDARD 17)

# Add the executable
add_executable(ProjectClangDoc1
        src/main.cpp
        src/Calculator.cpp
        src/Circle.cpp
        src/Rectangle.cpp
)

# Include directories
target_include_directories(ProjectClangDoc1 PUBLIC include)
