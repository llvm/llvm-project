# =================================================================================
# Lifetime Analysis Benchmarking Target
# =================================================================================
# This target allows running performance benchmarks for the clang lifetime analysis
# using a Python script (with managed dependencies).

find_package(Python3 COMPONENTS Interpreter REQUIRED)

# Define paths for the virtual environment and requirements file.
set(LIFETIME_BENCHMARK_SCRIPT 
  "${CMAKE_CURRENT_SOURCE_DIR}/benchmark.py")
set(LIFETIME_BENCHMARK_VENV_DIR "${CMAKE_CURRENT_BINARY_DIR}/benchmark-venv")
set(LIFETIME_BENCHMARK_REQUIREMENTS
  "${CMAKE_CURRENT_SOURCE_DIR}/requirements.txt")
set(LIFETIME_BENCHMARK_OUTPUT_DIR
  "${CMAKE_CURRENT_BINARY_DIR}/benchmark_results")


if(EXISTS ${LIFETIME_BENCHMARK_SCRIPT} AND EXISTS ${LIFETIME_BENCHMARK_REQUIREMENTS})

  # Set up the virtual environment and install packages
  add_custom_command(
    OUTPUT ${LIFETIME_BENCHMARK_VENV_DIR}/pyvenv.cfg
    COMMAND ${Python3_EXECUTABLE} -m venv ${LIFETIME_BENCHMARK_VENV_DIR}
    COMMAND ${LIFETIME_BENCHMARK_VENV_DIR}/bin/python -m pip install -r ${LIFETIME_BENCHMARK_REQUIREMENTS}
    DEPENDS ${LIFETIME_BENCHMARK_REQUIREMENTS}
    COMMENT "Creating Python virtual environment and installing dependencies for benchmark..."
  )
  add_custom_target(benchmark_venv_setup
    DEPENDS ${LIFETIME_BENCHMARK_VENV_DIR}/pyvenv.cfg
  )

  # Main benchmark target
  add_custom_target(benchmark_lifetime_safety_analysis
    COMMAND ${LIFETIME_BENCHMARK_VENV_DIR}/bin/python ${LIFETIME_BENCHMARK_SCRIPT}
            --clang-binary ${LLVM_BINARY_DIR}/bin/clang
            --output-dir ${LIFETIME_BENCHMARK_OUTPUT_DIR}

    DEPENDS clang benchmark_venv_setup

    # Display the output directly in the console.
    USES_TERMINAL
    
    COMMENT "Running Lifetime Analysis performance benchmarks..."
  )

  set_target_properties(benchmark_lifetime_safety_analysis 
    PROPERTIES FOLDER "Clang/Benchmarks")
endif()
