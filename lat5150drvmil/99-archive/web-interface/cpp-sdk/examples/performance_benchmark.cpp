/**
 * @file performance_benchmark.cpp
 * @brief Comprehensive performance benchmarking for DSMIL C++ SDK
 * 
 * This application provides detailed performance measurements and benchmarks
 * for various SDK operations and configurations.
 */

#include <dsmil/client.hpp>
#include <iostream>
#include <chrono>
#include <vector>
#include <thread>
#include <future>
#include <random>
#include <algorithm>
#include <iomanip>
#include <fstream>

class PerformanceBenchmark {
public:
    struct BenchmarkResult {
        std::string test_name;
        uint32_t operations;
        std::chrono::milliseconds total_time;
        std::chrono::milliseconds avg_latency;
        std::chrono::milliseconds p50_latency;
        std::chrono::milliseconds p95_latency;
        std::chrono::milliseconds p99_latency;
        std::chrono::milliseconds min_latency;
        std::chrono::milliseconds max_latency;
        double operations_per_second;
        double success_rate;
        uint32_t successful_operations;
        uint32_t failed_operations;
    };
    
    explicit PerformanceBenchmark(dsmil::Client& client) : client_(client) {}
    
    void run_all_benchmarks() {
        std::cout << "DSMIL C++ SDK Performance Benchmark Suite" << std::endl;
        std::cout << "==========================================" << std::endl << std::endl;
        
        results_.clear();
        
        // Single-threaded benchmarks
        results_.push_back(benchmark_single_read());
        results_.push_back(benchmark_single_write());
        results_.push_back(benchmark_bulk_read());
        
        // Multi-threaded benchmarks
        results_.push_back(benchmark_concurrent_reads());
        results_.push_back(benchmark_async_operations());
        
        // Memory and latency benchmarks
        results_.push_back(benchmark_latency_distribution());
        
        // Print summary
        print_summary();
        
        // Export results
        export_results_csv("benchmark_results.csv");
    }

private:
    dsmil::Client& client_;
    std::vector<BenchmarkResult> results_;
    
    BenchmarkResult benchmark_single_read() {
        std::cout << "Running single-threaded read benchmark..." << std::endl;
        
        const int num_operations = 1000;
        const dsmil::DeviceId test_device = 0x8000;
        
        std::vector<std::chrono::milliseconds> latencies;
        latencies.reserve(num_operations);
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        int successful = 0;
        int failed = 0;
        
        for (int i = 0; i < num_operations; ++i) {
            auto op_start = std::chrono::high_resolution_clock::now();
            
            try {
                auto result = client_.read_device_sync(test_device, dsmil::Register::STATUS);
                
                auto op_end = std::chrono::high_resolution_clock::now();
                auto latency = std::chrono::duration_cast<std::chrono::milliseconds>(op_end - op_start);
                latencies.push_back(latency);
                
                if (result.success) {
                    successful++;
                } else {
                    failed++;
                }
                
            } catch (const std::exception& e) {
                auto op_end = std::chrono::high_resolution_clock::now();
                auto latency = std::chrono::duration_cast<std::chrono::milliseconds>(op_end - op_start);
                latencies.push_back(latency);
                failed++;
            }
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto total_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        return calculate_result("Single-threaded Read", num_operations, total_time, latencies, successful, failed);
    }
    
    BenchmarkResult benchmark_single_write() {
        std::cout << "Running single-threaded write benchmark..." << std::endl;
        
        const int num_operations = 100;  // Fewer writes for safety
        const dsmil::DeviceId test_device = 0x8048;  // Auxiliary device
        
        std::vector<std::chrono::milliseconds> latencies;
        latencies.reserve(num_operations);
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        int successful = 0;
        int failed = 0;
        
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<uint32_t> dis(0x1000, 0xFFFF);
        
        for (int i = 0; i < num_operations; ++i) {
            auto op_start = std::chrono::high_resolution_clock::now();
            
            try {
                dsmil::WriteRequest write_req;
                write_req.device_id = test_device;
                write_req.register_type = dsmil::Register::DATA;
                write_req.offset = 0;
                write_req.data = dis(gen);
                write_req.justification = "Performance benchmark write test";
                
                auto result = client_.write_device_sync(write_req);
                
                auto op_end = std::chrono::high_resolution_clock::now();
                auto latency = std::chrono::duration_cast<std::chrono::milliseconds>(op_end - op_start);
                latencies.push_back(latency);
                
                if (result.success) {
                    successful++;
                } else {
                    failed++;
                }
                
            } catch (const std::exception& e) {
                auto op_end = std::chrono::high_resolution_clock::now();
                auto latency = std::chrono::duration_cast<std::chrono::milliseconds>(op_end - op_start);
                latencies.push_back(latency);
                failed++;
            }
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto total_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        return calculate_result("Single-threaded Write", num_operations, total_time, latencies, successful, failed);
    }
    
    BenchmarkResult benchmark_bulk_read() {
        std::cout << "Running bulk read benchmark..." << std::endl;
        
        const int num_operations = 100;  // Bulk operations
        const int devices_per_operation = 20;
        
        std::vector<dsmil::DeviceId> devices;
        for (int i = 0; i < devices_per_operation; ++i) {
            devices.push_back(0x8000 + i);
        }
        
        std::vector<std::chrono::milliseconds> latencies;
        latencies.reserve(num_operations);
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        int successful = 0;
        int failed = 0;
        int total_device_ops = 0;
        int successful_device_ops = 0;
        
        for (int i = 0; i < num_operations; ++i) {
            auto op_start = std::chrono::high_resolution_clock::now();
            
            try {
                auto result = client_.bulk_read_sync(devices, dsmil::Register::STATUS);
                
                auto op_end = std::chrono::high_resolution_clock::now();
                auto latency = std::chrono::duration_cast<std::chrono::milliseconds>(op_end - op_start);
                latencies.push_back(latency);
                
                if (result.overall_success) {
                    successful++;
                } else {
                    failed++;
                }
                
                total_device_ops += result.results.size();
                for (const auto& device_result : result.results) {
                    if (device_result.success) {
                        successful_device_ops++;
                    }
                }
                
            } catch (const std::exception& e) {
                auto op_end = std::chrono::high_resolution_clock::now();
                auto latency = std::chrono::duration_cast<std::chrono::milliseconds>(op_end - op_start);
                latencies.push_back(latency);
                failed++;
                total_device_ops += devices_per_operation;  // Assume all failed
            }
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto total_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        auto result = calculate_result("Bulk Read", num_operations, total_time, latencies, successful, failed);
        
        // Add bulk-specific metrics
        std::cout << "  Total device operations: " << total_device_ops << std::endl;
        std::cout << "  Successful device operations: " << successful_device_ops << std::endl;
        std::cout << "  Device success rate: " << std::fixed << std::setprecision(2) 
                  << (100.0 * successful_device_ops / total_device_ops) << "%" << std::endl;
        std::cout << "  Device operations/sec: " << std::fixed << std::setprecision(0) 
                  << (1000.0 * total_device_ops / total_time.count()) << std::endl;
        
        return result;
    }
    
    BenchmarkResult benchmark_concurrent_reads() {
        std::cout << "Running concurrent read benchmark..." << std::endl;
        
        const int num_threads = 4;
        const int operations_per_thread = 250;
        const int total_operations = num_threads * operations_per_thread;
        
        std::vector<std::future<std::vector<std::chrono::milliseconds>>> futures;
        std::vector<std::future<std::pair<int, int>>> result_futures;  // success, failed
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // Launch concurrent threads
        for (int t = 0; t < num_threads; ++t) {
            // Latency tracking future
            futures.push_back(std::async(std::launch::async, [this, operations_per_thread, t]() {
                std::vector<std::chrono::milliseconds> thread_latencies;
                thread_latencies.reserve(operations_per_thread);
                
                const dsmil::DeviceId device_id = 0x8000 + (t % 12);  // Distribute across devices
                
                for (int i = 0; i < operations_per_thread; ++i) {
                    auto op_start = std::chrono::high_resolution_clock::now();
                    
                    try {
                        client_.read_device_sync(device_id, dsmil::Register::STATUS);
                        auto op_end = std::chrono::high_resolution_clock::now();
                        auto latency = std::chrono::duration_cast<std::chrono::milliseconds>(op_end - op_start);
                        thread_latencies.push_back(latency);
                    } catch (const std::exception& e) {
                        auto op_end = std::chrono::high_resolution_clock::now();
                        auto latency = std::chrono::duration_cast<std::chrono::milliseconds>(op_end - op_start);
                        thread_latencies.push_back(latency);
                    }
                }
                
                return thread_latencies;
            }));
            
            // Result tracking future
            result_futures.push_back(std::async(std::launch::async, [this, operations_per_thread, t]() {
                int successful = 0;
                int failed = 0;
                
                const dsmil::DeviceId device_id = 0x8000 + (t % 12);
                
                for (int i = 0; i < operations_per_thread; ++i) {
                    try {
                        auto result = client_.read_device_sync(device_id, dsmil::Register::STATUS);
                        if (result.success) {
                            successful++;
                        } else {
                            failed++;
                        }
                    } catch (const std::exception& e) {
                        failed++;
                    }
                }
                
                return std::make_pair(successful, failed);
            }));
        }
        
        // Collect results
        std::vector<std::chrono::milliseconds> all_latencies;
        int total_successful = 0;
        int total_failed = 0;
        
        for (auto& future : futures) {
            auto thread_latencies = future.get();
            all_latencies.insert(all_latencies.end(), thread_latencies.begin(), thread_latencies.end());
        }
        
        for (auto& future : result_futures) {
            auto results = future.get();
            total_successful += results.first;
            total_failed += results.second;
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto total_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        return calculate_result("Concurrent Read", total_operations, total_time, all_latencies, 
                               total_successful, total_failed);
    }
    
    BenchmarkResult benchmark_async_operations() {
        std::cout << "Running async operations benchmark..." << std::endl;
        
        const int num_operations = 500;
        std::vector<std::chrono::milliseconds> latencies;
        latencies.reserve(num_operations);
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // Launch async operations
        std::vector<std::future<dsmil::DeviceResult>> futures;
        for (int i = 0; i < num_operations; ++i) {
            const dsmil::DeviceId device_id = 0x8000 + (i % 12);
            futures.push_back(client_.read_device_async(device_id, dsmil::Register::STATUS));
        }
        
        // Wait for all operations to complete
        int successful = 0;
        int failed = 0;
        
        for (int i = 0; i < num_operations; ++i) {
            auto op_start = std::chrono::high_resolution_clock::now();
            
            try {
                auto result = futures[i].get();
                auto op_end = std::chrono::high_resolution_clock::now();
                auto latency = std::chrono::duration_cast<std::chrono::milliseconds>(op_end - op_start);
                latencies.push_back(latency);
                
                if (result.success) {
                    successful++;
                } else {
                    failed++;
                }
                
            } catch (const std::exception& e) {
                auto op_end = std::chrono::high_resolution_clock::now();
                auto latency = std::chrono::duration_cast<std::chrono::milliseconds>(op_end - op_start);
                latencies.push_back(latency);
                failed++;
            }
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto total_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        return calculate_result("Async Operations", num_operations, total_time, latencies, successful, failed);
    }
    
    BenchmarkResult benchmark_latency_distribution() {
        std::cout << "Running latency distribution analysis..." << std::endl;
        
        const int num_operations = 2000;  // Larger sample for distribution analysis
        const dsmil::DeviceId test_device = 0x8000;
        
        std::vector<std::chrono::milliseconds> latencies;
        latencies.reserve(num_operations);
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        int successful = 0;
        int failed = 0;
        
        for (int i = 0; i < num_operations; ++i) {
            auto op_start = std::chrono::high_resolution_clock::now();
            
            try {
                auto result = client_.read_device_sync(test_device, dsmil::Register::STATUS);
                
                auto op_end = std::chrono::high_resolution_clock::now();
                auto latency = std::chrono::duration_cast<std::chrono::milliseconds>(op_end - op_start);
                latencies.push_back(latency);
                
                if (result.success) {
                    successful++;
                } else {
                    failed++;
                }
                
            } catch (const std::exception& e) {
                auto op_end = std::chrono::high_resolution_clock::now();
                auto latency = std::chrono::duration_cast<std::chrono::milliseconds>(op_end - op_start);
                latencies.push_back(latency);
                failed++;
            }
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto total_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        // Print detailed latency distribution
        std::sort(latencies.begin(), latencies.end());
        
        std::cout << "  Latency distribution:" << std::endl;
        std::cout << "    P1:  " << latencies[num_operations * 1 / 100].count() << "ms" << std::endl;
        std::cout << "    P5:  " << latencies[num_operations * 5 / 100].count() << "ms" << std::endl;
        std::cout << "    P10: " << latencies[num_operations * 10 / 100].count() << "ms" << std::endl;
        std::cout << "    P25: " << latencies[num_operations * 25 / 100].count() << "ms" << std::endl;
        std::cout << "    P50: " << latencies[num_operations * 50 / 100].count() << "ms" << std::endl;
        std::cout << "    P75: " << latencies[num_operations * 75 / 100].count() << "ms" << std::endl;
        std::cout << "    P90: " << latencies[num_operations * 90 / 100].count() << "ms" << std::endl;
        std::cout << "    P95: " << latencies[num_operations * 95 / 100].count() << "ms" << std::endl;
        std::cout << "    P99: " << latencies[num_operations * 99 / 100].count() << "ms" << std::endl;
        
        return calculate_result("Latency Distribution", num_operations, total_time, latencies, successful, failed);
    }
    
    BenchmarkResult calculate_result(const std::string& test_name, int num_operations,
                                   std::chrono::milliseconds total_time,
                                   const std::vector<std::chrono::milliseconds>& latencies,
                                   int successful, int failed) {
        BenchmarkResult result;
        result.test_name = test_name;
        result.operations = num_operations;
        result.total_time = total_time;
        result.successful_operations = successful;
        result.failed_operations = failed;
        result.success_rate = static_cast<double>(successful) / num_operations;
        result.operations_per_second = 1000.0 * num_operations / total_time.count();
        
        // Calculate latency statistics
        auto sorted_latencies = latencies;
        std::sort(sorted_latencies.begin(), sorted_latencies.end());
        
        if (!sorted_latencies.empty()) {
            result.min_latency = sorted_latencies.front();
            result.max_latency = sorted_latencies.back();
            result.p50_latency = sorted_latencies[sorted_latencies.size() * 50 / 100];
            result.p95_latency = sorted_latencies[sorted_latencies.size() * 95 / 100];
            result.p99_latency = sorted_latencies[sorted_latencies.size() * 99 / 100];
            
            // Calculate average
            long long total_latency = 0;
            for (const auto& latency : sorted_latencies) {
                total_latency += latency.count();
            }
            result.avg_latency = std::chrono::milliseconds(total_latency / sorted_latencies.size());
        }
        
        // Print results
        print_result(result);
        
        return result;
    }
    
    void print_result(const BenchmarkResult& result) {
        std::cout << "Results for " << result.test_name << ":" << std::endl;
        std::cout << "  Operations: " << result.operations << std::endl;
        std::cout << "  Total time: " << result.total_time.count() << "ms" << std::endl;
        std::cout << "  Success rate: " << std::fixed << std::setprecision(2) 
                  << result.success_rate * 100.0 << "%" << std::endl;
        std::cout << "  Operations/sec: " << std::fixed << std::setprecision(0) 
                  << result.operations_per_second << std::endl;
        std::cout << "  Average latency: " << result.avg_latency.count() << "ms" << std::endl;
        std::cout << "  P50 latency: " << result.p50_latency.count() << "ms" << std::endl;
        std::cout << "  P95 latency: " << result.p95_latency.count() << "ms" << std::endl;
        std::cout << "  P99 latency: " << result.p99_latency.count() << "ms" << std::endl;
        std::cout << "  Min latency: " << result.min_latency.count() << "ms" << std::endl;
        std::cout << "  Max latency: " << result.max_latency.count() << "ms" << std::endl;
        std::cout << std::endl;
    }
    
    void print_summary() {
        std::cout << "=== BENCHMARK SUMMARY ===" << std::endl << std::endl;
        
        std::cout << std::left << std::setw(25) << "Test Name" 
                  << std::setw(12) << "Ops/sec" 
                  << std::setw(12) << "Success%" 
                  << std::setw(12) << "Avg ms" 
                  << std::setw(12) << "P95 ms" 
                  << std::setw(12) << "P99 ms" << std::endl;
        std::cout << std::string(85, '-') << std::endl;
        
        for (const auto& result : results_) {
            std::cout << std::left << std::setw(25) << result.test_name
                      << std::setw(12) << std::fixed << std::setprecision(0) << result.operations_per_second
                      << std::setw(12) << std::fixed << std::setprecision(1) << result.success_rate * 100.0
                      << std::setw(12) << result.avg_latency.count()
                      << std::setw(12) << result.p95_latency.count()
                      << std::setw(12) << result.p99_latency.count() << std::endl;
        }
        
        std::cout << std::endl;
    }
    
    void export_results_csv(const std::string& filename) {
        std::cout << "Exporting results to " << filename << "..." << std::endl;
        
        std::ofstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Failed to open " << filename << " for writing" << std::endl;
            return;
        }
        
        // CSV header
        file << "Test Name,Operations,Total Time (ms),Success Rate,Operations/sec,"
             << "Avg Latency (ms),P50 Latency (ms),P95 Latency (ms),P99 Latency (ms),"
             << "Min Latency (ms),Max Latency (ms),Successful Ops,Failed Ops" << std::endl;
        
        // CSV data
        for (const auto& result : results_) {
            file << result.test_name << ","
                 << result.operations << ","
                 << result.total_time.count() << ","
                 << std::fixed << std::setprecision(4) << result.success_rate << ","
                 << std::fixed << std::setprecision(2) << result.operations_per_second << ","
                 << result.avg_latency.count() << ","
                 << result.p50_latency.count() << ","
                 << result.p95_latency.count() << ","
                 << result.p99_latency.count() << ","
                 << result.min_latency.count() << ","
                 << result.max_latency.count() << ","
                 << result.successful_operations << ","
                 << result.failed_operations << std::endl;
        }
        
        file.close();
        std::cout << "Results exported successfully!" << std::endl;
    }
};

int main() {
    try {
        // Configure client for performance testing
        dsmil::ClientConfig config;
        config.base_url = "https://dsmil-control.mil";
        config.api_version = "2.0";
        config.connection_pool_size = 20;  // Larger pool for concurrent tests
        config.enable_metrics = true;
        config.kernel_module_path = "/dev/dsmil_control";  // Enable kernel module if available
        
        dsmil::Client client(config);
        
        std::cout << "Authenticating for performance testing..." << std::endl;
        auto auth_result = client.authenticate("perf_user", "perf_pass", dsmil::ClientType::Cpp);
        
        if (!auth_result.success) {
            std::cerr << "Authentication failed: " << auth_result.error_message << std::endl;
            return 1;
        }
        
        std::cout << "Authentication successful!" << std::endl << std::endl;
        
        // Run benchmarks
        PerformanceBenchmark benchmark(client);
        benchmark.run_all_benchmarks();
        
        // Show final SDK metrics
        std::cout << "=== FINAL SDK METRICS ===" << std::endl;
        auto metrics = client.get_performance_metrics();
        std::cout << "Total operations: " << metrics.total_operations << std::endl;
        std::cout << "Successful operations: " << metrics.successful_operations << std::endl;
        std::cout << "Failed operations: " << metrics.failed_operations << std::endl;
        std::cout << "Overall success rate: " << std::fixed << std::setprecision(2) 
                  << metrics.success_rate * 100.0 << "%" << std::endl;
        std::cout << "Overall average latency: " << metrics.avg_latency_ms.count() << "ms" << std::endl;
        std::cout << "Overall P95 latency: " << metrics.p95_latency_ms.count() << "ms" << std::endl;
        std::cout << "Overall P99 latency: " << metrics.p99_latency_ms.count() << "ms" << std::endl;
        
        std::cout << std::endl << "Performance benchmark completed successfully!" << std::endl;
        
    } catch (const dsmil::DSMILException& e) {
        std::cerr << "DSMIL SDK error: " << e.what() << std::endl;
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "Unexpected error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}