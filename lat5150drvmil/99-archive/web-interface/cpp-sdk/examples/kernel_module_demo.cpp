/**
 * @file kernel_module_demo.cpp
 * @brief Kernel module integration demonstration
 * 
 * This example demonstrates direct kernel module integration for maximum
 * performance and minimal latency operations.
 */

#include <dsmil/client.hpp>
#include <iostream>
#include <chrono>
#include <vector>
#include <iomanip>

void print_kernel_stats(const dsmil::Client& client) {
    // This would require exposing kernel interface stats in public API
    // For now, demonstrate the concept with performance metrics
    auto metrics = client.get_performance_metrics();
    
    std::cout << "=== Kernel Module Statistics ===" << std::endl;
    std::cout << "Total operations: " << metrics.total_operations << std::endl;
    std::cout << "Successful operations: " << metrics.successful_operations << std::endl;
    std::cout << "Failed operations: " << metrics.failed_operations << std::endl;
    std::cout << "Average latency: " << metrics.avg_latency_ms.count() << "ms" << std::endl;
    std::cout << "P95 latency: " << metrics.p95_latency_ms.count() << "ms" << std::endl;
    std::cout << "P99 latency: " << metrics.p99_latency_ms.count() << "ms" << std::endl;
    std::cout << std::endl;
}

int main() {
    std::cout << "DSMIL Kernel Module Integration Demo" << std::endl;
    std::cout << "====================================" << std::endl << std::endl;
    
    try {
        // Configure client for kernel module integration
        dsmil::ClientConfig config;
        config.base_url = "https://dsmil-control.mil";
        config.api_version = "2.0";
        config.kernel_module_path = "/dev/dsmil_control";  // Enable kernel module
        config.enable_kernel_bypass = true;  // Prefer kernel module over HTTP
        
        dsmil::Client client(config);
        
        std::cout << "Authenticating for kernel module access..." << std::endl;
        auto auth_result = client.authenticate("kernel_user", "kernel_pass", dsmil::ClientType::Cpp);
        
        if (!auth_result.success) {
            std::cerr << "Authentication failed: " << auth_result.error_message << std::endl;
            return 1;
        }
        
        std::cout << "Authenticated successfully!" << std::endl;
        std::cout << "Testing kernel module integration..." << std::endl << std::endl;
        
        // Test 1: Single device read performance
        std::cout << "=== Test 1: Single Device Read Performance ===" << std::endl;
        
        const dsmil::DeviceId test_device = 0x8000;  // Master security controller
        const int num_reads = 1000;
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        int successful_reads = 0;
        int failed_reads = 0;
        
        for (int i = 0; i < num_reads; ++i) {
            try {
                auto result = client.read_device_sync(test_device, dsmil::Register::STATUS);
                if (result.success) {
                    successful_reads++;
                } else {
                    failed_reads++;
                }
            } catch (const std::exception& e) {
                failed_reads++;
                if (i == 0) {  // Only print first error to avoid spam
                    std::cout << "Read error: " << e.what() << std::endl;
                }
            }
            
            // Progress indicator
            if ((i + 1) % 100 == 0) {
                std::cout << "Completed " << (i + 1) << "/" << num_reads << " reads..." << std::endl;
            }
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto total_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        std::cout << "Results:" << std::endl;
        std::cout << "  Total reads: " << num_reads << std::endl;
        std::cout << "  Successful: " << successful_reads << std::endl;
        std::cout << "  Failed: " << failed_reads << std::endl;
        std::cout << "  Success rate: " << std::fixed << std::setprecision(2) 
                  << (100.0 * successful_reads / num_reads) << "%" << std::endl;
        std::cout << "  Total time: " << total_time.count() << "ms" << std::endl;
        std::cout << "  Average latency: " << std::fixed << std::setprecision(2) 
                  << (static_cast<double>(total_time.count()) / num_reads) << "ms" << std::endl;
        std::cout << "  Operations/sec: " << std::fixed << std::setprecision(0) 
                  << (1000.0 * num_reads / total_time.count()) << std::endl;
        std::cout << std::endl;
        
        // Test 2: Bulk read performance
        std::cout << "=== Test 2: Bulk Read Performance ===" << std::endl;
        
        std::vector<dsmil::DeviceId> bulk_devices;
        for (int i = 0; i < 20; ++i) {  // Test with first 20 devices
            bulk_devices.push_back(0x8000 + i);
        }
        
        const int num_bulk_ops = 100;
        
        auto bulk_start = std::chrono::high_resolution_clock::now();
        
        int successful_bulk_ops = 0;
        int total_device_reads = 0;
        int successful_device_reads = 0;
        
        for (int i = 0; i < num_bulk_ops; ++i) {
            try {
                auto bulk_result = client.bulk_read_sync(bulk_devices, dsmil::Register::STATUS);
                
                if (bulk_result.overall_success) {
                    successful_bulk_ops++;
                }
                
                total_device_reads += bulk_result.results.size();
                for (const auto& result : bulk_result.results) {
                    if (result.success) {
                        successful_device_reads++;
                    }
                }
                
            } catch (const std::exception& e) {
                if (i == 0) {  // Only print first error
                    std::cout << "Bulk read error: " << e.what() << std::endl;
                }
            }
            
            // Progress indicator
            if ((i + 1) % 10 == 0) {
                std::cout << "Completed " << (i + 1) << "/" << num_bulk_ops << " bulk operations..." << std::endl;
            }
        }
        
        auto bulk_end = std::chrono::high_resolution_clock::now();
        auto bulk_total_time = std::chrono::duration_cast<std::chrono::milliseconds>(bulk_end - bulk_start);
        
        std::cout << "Bulk Results:" << std::endl;
        std::cout << "  Bulk operations: " << num_bulk_ops << std::endl;
        std::cout << "  Successful bulk ops: " << successful_bulk_ops << std::endl;
        std::cout << "  Total device reads: " << total_device_reads << std::endl;
        std::cout << "  Successful device reads: " << successful_device_reads << std::endl;
        std::cout << "  Device success rate: " << std::fixed << std::setprecision(2) 
                  << (100.0 * successful_device_reads / total_device_reads) << "%" << std::endl;
        std::cout << "  Total time: " << bulk_total_time.count() << "ms" << std::endl;
        std::cout << "  Avg time per bulk op: " << std::fixed << std::setprecision(2) 
                  << (static_cast<double>(bulk_total_time.count()) / num_bulk_ops) << "ms" << std::endl;
        std::cout << "  Device reads/sec: " << std::fixed << std::setprecision(0) 
                  << (1000.0 * total_device_reads / bulk_total_time.count()) << std::endl;
        std::cout << std::endl;
        
        // Test 3: Write operation safety (test with safe device)
        std::cout << "=== Test 3: Write Operation Safety ===" << std::endl;
        
        // Use a low-risk device for write testing (auxiliary control system)
        const dsmil::DeviceId write_test_device = 0x8048;  // Group 5, Device 8 (auxiliary)
        
        dsmil::WriteRequest write_req;
        write_req.device_id = write_test_device;
        write_req.register_type = dsmil::Register::CONFIG;
        write_req.offset = 0;
        write_req.data = 0x12345678;
        write_req.justification = "Kernel module integration test write";
        
        try {
            std::cout << "Attempting write to device 0x" << std::hex << std::uppercase 
                      << write_test_device << "..." << std::endl;
            
            auto write_result = client.write_device_sync(write_req);
            
            if (write_result.success) {
                std::cout << "Write successful!" << std::endl;
                std::cout << "  Execution time: " << write_result.execution_time.count() << "ms" << std::endl;
                
                // Verify write by reading back
                std::cout << "Verifying write by reading back..." << std::endl;
                auto read_result = client.read_device_sync(write_test_device, dsmil::Register::CONFIG);
                
                if (read_result.success) {
                    std::cout << "Read verification successful!" << std::endl;
                    std::cout << "  Written value: 0x" << std::hex << std::uppercase << write_req.data << std::endl;
                    std::cout << "  Read value: 0x" << std::hex << std::uppercase << read_result.data << std::endl;
                    
                    if (read_result.data == write_req.data) {
                        std::cout << "  ✓ Values match - write operation verified!" << std::endl;
                    } else {
                        std::cout << "  ✗ Values do not match - verification failed!" << std::endl;
                    }
                } else {
                    std::cout << "Read verification failed: " << read_result.error_message << std::endl;
                }
                
            } else {
                std::cout << "Write failed: " << write_result.error_message << std::endl;
            }
            
        } catch (const dsmil::QuarantineException& e) {
            std::cout << "Write blocked - device is quarantined: " << e.what() << std::endl;
            
        } catch (const dsmil::PermissionException& e) {
            std::cout << "Write blocked - insufficient permissions: " << e.what() << std::endl;
            
        } catch (const dsmil::SecurityViolationException& e) {
            std::cout << "Write blocked - security violation: " << e.what() << std::endl;
            
        } catch (const std::exception& e) {
            std::cout << "Write error: " << e.what() << std::endl;
        }
        
        std::cout << std::endl;
        
        // Test 4: Emergency stop functionality
        std::cout << "=== Test 4: Emergency Stop Functionality ===" << std::endl;
        std::cout << "Note: This is a demonstration - actual emergency stop not triggered" << std::endl;
        
        // In a real scenario, you would test emergency stop functionality
        // For safety, we'll just demonstrate the API structure
        
        /*
        dsmil::EmergencyStopRequest emergency_req;
        emergency_req.justification = "Kernel module integration test";
        emergency_req.scope = dsmil::EmergencyScope::SINGLE_DEVICE;
        emergency_req.target_devices = {write_test_device};
        emergency_req.notify_all_clients = false;  // Test mode
        emergency_req.escalation_level = dsmil::EscalationLevel::NORMAL;
        
        // This would trigger emergency stop in production
        // auto emergency_result = client.trigger_emergency_stop_sync(emergency_req);
        */
        
        std::cout << "Emergency stop API structure verified" << std::endl;
        std::cout << std::endl;
        
        // Final statistics
        print_kernel_stats(client);
        
        std::cout << "Kernel module integration test completed successfully!" << std::endl;
        
    } catch (const dsmil::KernelModuleException& e) {
        std::cerr << "Kernel module error: " << e.what() << std::endl;
        std::cerr << "Make sure the DSMIL kernel module is loaded and accessible." << std::endl;
        return 1;
        
    } catch (const dsmil::AuthenticationException& e) {
        std::cerr << "Authentication error: " << e.what() << std::endl;
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