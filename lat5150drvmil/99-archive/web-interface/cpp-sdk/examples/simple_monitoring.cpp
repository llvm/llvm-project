/**
 * @file simple_monitoring.cpp
 * @brief Simple device monitoring application using DSMIL C++ SDK
 * 
 * This example demonstrates basic usage of the DSMIL C++ SDK for monitoring
 * critical devices in real-time.
 */

#include <dsmil/client.hpp>
#include <iostream>
#include <thread>
#include <chrono>
#include <csignal>
#include <atomic>
#include <iomanip>

std::atomic<bool> running{true};

void signal_handler(int signal) {
    std::cout << "\nShutdown signal received (" << signal << ")..." << std::endl;
    running = false;
}

void print_device_status(const dsmil::DeviceResult& result) {
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "[" << std::put_time(std::localtime(&std::chrono::system_clock::to_time_t(result.timestamp)), "%H:%M:%S") << "] ";
    std::cout << "Device 0x" << std::hex << std::uppercase << result.device_id << ": ";
    
    if (result.success) {
        std::cout << "Status=0x" << result.data;
        
        // Interpret common status bits
        if (result.data & 0x80000000) {
            std::cout << " [ERROR]";
        } else if (result.data & 0x40000000) {
            std::cout << " [WARNING]";
        } else {
            std::cout << " [OK]";
        }
        
        std::cout << " (took " << result.execution_time.count() << "ms)";
    } else {
        std::cout << "FAILED: " << result.error_message;
    }
    
    std::cout << std::dec << std::endl;
}

int main(int argc, char* argv[]) {
    // Setup signal handling
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    
    std::cout << "DSMIL Simple Monitoring Application v2.0.1" << std::endl;
    std::cout << "==========================================" << std::endl << std::endl;
    
    try {
        // Initialize client with default configuration
        dsmil::Client client("https://dsmil-control.mil", "2.0");
        
        std::cout << "Authenticating..." << std::endl;
        
        // Authenticate (use environment variables or command line args in production)
        std::string username = "monitor";
        std::string password = "secure_pass";
        
        auto auth_result = client.authenticate(username, password, dsmil::ClientType::Cpp);
        
        if (!auth_result.success) {
            std::cerr << "Authentication failed: " << auth_result.error_message << std::endl;
            return 1;
        }
        
        std::cout << "Authenticated as: " << auth_result.user_context.username 
                  << " (clearance: ";
        
        switch (auth_result.user_context.clearance_level) {
            case dsmil::ClearanceLevel::RESTRICTED: std::cout << "RESTRICTED"; break;
            case dsmil::ClearanceLevel::CONFIDENTIAL: std::cout << "CONFIDENTIAL"; break;
            case dsmil::ClearanceLevel::SECRET: std::cout << "SECRET"; break;
            case dsmil::ClearanceLevel::TOP_SECRET: std::cout << "TOP_SECRET"; break;
            case dsmil::ClearanceLevel::SCI: std::cout << "SCI"; break;
        }
        
        std::cout << ")" << std::endl;
        std::cout << "Authorized devices: " << auth_result.user_context.authorized_devices.size() << std::endl;
        std::cout << std::endl;
        
        // Get list of critical devices to monitor
        std::vector<dsmil::DeviceId> critical_devices;
        
        // Add master security controllers (Group 0)
        for (int i = 0; i < 12; ++i) {
            critical_devices.push_back(0x8000 + i);
        }
        
        // Add primary system interfaces (Group 1)
        for (int i = 12; i < 24; ++i) {
            critical_devices.push_back(0x8000 + i);
        }
        
        std::cout << "Monitoring " << critical_devices.size() << " critical devices..." << std::endl;
        std::cout << "Press Ctrl+C to stop monitoring" << std::endl << std::endl;
        
        // Monitor devices in a loop
        uint32_t iteration = 0;
        while (running) {
            ++iteration;
            std::cout << "=== Monitoring Cycle #" << iteration << " ===" << std::endl;
            
            auto cycle_start = std::chrono::high_resolution_clock::now();
            
            // Read all critical devices
            for (dsmil::DeviceId device_id : critical_devices) {
                if (!running) break;
                
                try {
                    auto result = client.read_device_sync(device_id, dsmil::Register::STATUS);
                    print_device_status(result);
                    
                    // Alert on critical conditions
                    if (result.success && (result.data & 0x80000000)) {
                        std::cout << "*** ALERT: Device 0x" << std::hex << std::uppercase 
                                  << device_id << " reports ERROR status! ***" << std::endl;
                    }
                    
                } catch (const dsmil::QuarantineException& e) {
                    std::cout << "Device 0x" << std::hex << std::uppercase << device_id 
                              << ": QUARANTINED - " << e.what() << std::endl;
                    
                } catch (const dsmil::PermissionException& e) {
                    std::cout << "Device 0x" << std::hex << std::uppercase << device_id 
                              << ": ACCESS DENIED - " << e.what() << std::endl;
                    
                } catch (const dsmil::DSMILException& e) {
                    std::cout << "Device 0x" << std::hex << std::uppercase << device_id 
                              << ": ERROR - " << e.what() << std::endl;
                }
            }
            
            auto cycle_end = std::chrono::high_resolution_clock::now();
            auto cycle_time = std::chrono::duration_cast<std::chrono::milliseconds>(cycle_end - cycle_start);
            
            std::cout << "Cycle completed in " << cycle_time.count() << "ms" << std::endl;
            
            // Show performance metrics every 10 cycles
            if (iteration % 10 == 0) {
                try {
                    auto metrics = client.get_performance_metrics();
                    std::cout << std::endl << "=== Performance Metrics ===" << std::endl;
                    std::cout << "Average latency: " << metrics.avg_latency_ms.count() << "ms" << std::endl;
                    std::cout << "Success rate: " << std::fixed << std::setprecision(1) 
                              << metrics.success_rate * 100.0 << "%" << std::endl;
                    std::cout << "Operations/sec: " << std::fixed << std::setprecision(1) 
                              << metrics.operations_per_second << std::endl;
                    std::cout << "Total operations: " << metrics.total_operations << std::endl;
                } catch (const std::exception& e) {
                    std::cout << "Failed to get metrics: " << e.what() << std::endl;
                }
            }
            
            std::cout << std::endl;
            
            // Wait before next monitoring cycle (adjust based on requirements)
            if (running) {
                std::this_thread::sleep_for(std::chrono::seconds(10));
            }
        }
        
        std::cout << "Monitoring stopped." << std::endl;
        
        // Final performance summary
        try {
            auto final_metrics = client.get_performance_metrics();
            std::cout << std::endl << "=== Final Performance Summary ===" << std::endl;
            std::cout << "Total operations: " << final_metrics.total_operations << std::endl;
            std::cout << "Successful operations: " << final_metrics.successful_operations << std::endl;
            std::cout << "Failed operations: " << final_metrics.failed_operations << std::endl;
            std::cout << "Overall success rate: " << std::fixed << std::setprecision(2) 
                      << final_metrics.success_rate * 100.0 << "%" << std::endl;
            std::cout << "Average latency: " << final_metrics.avg_latency_ms.count() << "ms" << std::endl;
            std::cout << "P95 latency: " << final_metrics.p95_latency_ms.count() << "ms" << std::endl;
        } catch (const std::exception& e) {
            std::cout << "Failed to get final metrics: " << e.what() << std::endl;
        }
        
    } catch (const dsmil::AuthenticationException& e) {
        std::cerr << "Authentication error: " << e.what() << std::endl;
        return 1;
        
    } catch (const dsmil::NetworkException& e) {
        std::cerr << "Network error: " << e.what() << std::endl;
        return 1;
        
    } catch (const dsmil::DSMILException& e) {
        std::cerr << "DSMIL SDK error: " << e.what() << std::endl;
        return 1;
        
    } catch (const std::exception& e) {
        std::cerr << "Unexpected error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}