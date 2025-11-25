cmake_minimum_required(VERSION 3.20)

# Find required packages
find_package(Threads REQUIRED)

# Example applications
add_executable(simple_monitoring simple_monitoring.cpp)
target_link_libraries(simple_monitoring 
    DSMILClient::DSMILClient 
    Threads::Threads
)

add_executable(high_performance_data_acquisition high_performance_data_acquisition.cpp)
target_link_libraries(high_performance_data_acquisition 
    DSMILClient::DSMILClient 
    Threads::Threads
)

add_executable(emergency_response_system emergency_response_system.cpp)
target_link_libraries(emergency_response_system 
    DSMILClient::DSMILClient 
    Threads::Threads
)

add_executable(bulk_operations_demo bulk_operations_demo.cpp)
target_link_libraries(bulk_operations_demo 
    DSMILClient::DSMILClient 
    Threads::Threads
)

add_executable(websocket_streaming_demo websocket_streaming_demo.cpp)
target_link_libraries(websocket_streaming_demo 
    DSMILClient::DSMILClient 
    Threads::Threads
)

add_executable(kernel_module_demo kernel_module_demo.cpp)
target_link_libraries(kernel_module_demo 
    DSMILClient::DSMILClient 
    Threads::Threads
)

add_executable(security_demo security_demo.cpp)
target_link_libraries(security_demo 
    DSMILClient::DSMILClient 
    Threads::Threads
)

# Performance benchmark
add_executable(performance_benchmark performance_benchmark.cpp)
target_link_libraries(performance_benchmark 
    DSMILClient::DSMILClient 
    Threads::Threads
)

# Install examples
install(TARGETS 
    simple_monitoring
    high_performance_data_acquisition
    emergency_response_system
    bulk_operations_demo
    websocket_streaming_demo
    kernel_module_demo
    security_demo
    performance_benchmark
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}/dsmil-examples
)