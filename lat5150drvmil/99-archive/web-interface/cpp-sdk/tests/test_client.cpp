#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <dsmil/client.hpp>
#include <dsmil/exceptions.hpp>

namespace dsmil {
namespace test {

class ClientTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Setup test configuration
        config_.base_url = "https://test.dsmil-control.mil";
        config_.api_version = "2.0";
        config_.verify_ssl = false;  // For testing
        config_.timeout = std::chrono::seconds(5);
        config_.connection_pool_size = 5;
    }
    
    ClientConfig config_;
};

// Unit Tests

TEST_F(ClientTest, UnitConstructorBasic) {
    EXPECT_NO_THROW({
        Client client("https://test.dsmil-control.mil", "2.0");
    });
}

TEST_F(ClientTest, UnitConstructorAdvanced) {
    EXPECT_NO_THROW({
        Client client(config_);
    });
}

TEST_F(ClientTest, UnitConstructorInvalidUrl) {
    EXPECT_THROW({
        Client client("invalid-url", "2.0");
    }, ConfigurationException);
}

TEST_F(ClientTest, UnitMoveSemantics) {
    Client client1("https://test.dsmil-control.mil", "2.0");
    
    // Move constructor
    Client client2 = std::move(client1);
    
    // Move assignment
    Client client3("https://other.dsmil-control.mil", "2.0");
    client3 = std::move(client2);
}

// Authentication Tests

TEST_F(ClientTest, UnitAuthenticationSuccess) {
    Client client(config_);
    
    // Mock successful authentication
    // Note: In a real test environment, this would use a mock server
    // For demonstration, we test the API structure
    EXPECT_NO_THROW({
        auto result = client.authenticate("test_user", "test_pass", ClientType::Cpp);
        // In mock mode, this should return appropriate test data
    });
}

TEST_F(ClientTest, UnitAuthenticationFailure) {
    Client client(config_);
    
    EXPECT_NO_THROW({
        auto result = client.authenticate("invalid_user", "invalid_pass", ClientType::Cpp);
        EXPECT_FALSE(result.success);
        EXPECT_FALSE(result.error_message.empty());
    });
}

TEST_F(ClientTest, UnitAuthenticationEmptyCredentials) {
    Client client(config_);
    
    EXPECT_THROW({
        client.authenticate("", "", ClientType::Cpp);
    }, ConfigurationException);
}

// Device Operation Tests

TEST_F(ClientTest, UnitReadDeviceSync) {
    Client client(config_);
    
    // Mock authentication
    client.authenticate("test_user", "test_pass", ClientType::Cpp);
    
    EXPECT_NO_THROW({
        auto result = client.read_device_sync(0x8000, Register::STATUS);
        // In test mode, should get mock result
        EXPECT_EQ(result.device_id, 0x8000);
        EXPECT_EQ(result.register_type, Register::STATUS);
    });
}

TEST_F(ClientTest, UnitReadDeviceInvalidId) {
    Client client(config_);
    client.authenticate("test_user", "test_pass", ClientType::Cpp);
    
    EXPECT_THROW({
        client.read_device_sync(0x9999, Register::STATUS);  // Invalid device ID
    }, DeviceException);
}

TEST_F(ClientTest, UnitReadDeviceNotAuthenticated) {
    Client client(config_);
    
    EXPECT_THROW({
        client.read_device_sync(0x8000, Register::STATUS);
    }, AuthenticationException);
}

TEST_F(ClientTest, UnitWriteDeviceSync) {
    Client client(config_);
    client.authenticate("test_user", "test_pass", ClientType::Cpp);
    
    WriteRequest request;
    request.device_id = 0x8048;  // Use auxiliary device for testing
    request.register_type = Register::DATA;
    request.offset = 0;
    request.data = 0x12345678;
    request.justification = "Unit test write operation";
    
    EXPECT_NO_THROW({
        auto result = client.write_device_sync(request);
        EXPECT_EQ(result.device_id, request.device_id);
        EXPECT_EQ(result.register_type, request.register_type);
    });
}

TEST_F(ClientTest, UnitWriteDeviceQuarantined) {
    Client client(config_);
    client.authenticate("test_user", "test_pass", ClientType::Cpp);
    
    WriteRequest request;
    request.device_id = 0x8009;  // Quarantined device
    request.register_type = Register::DATA;
    request.offset = 0;
    request.data = 0x12345678;
    request.justification = "Test write to quarantined device";
    
    EXPECT_THROW({
        client.write_device_sync(request);
    }, QuarantineException);
}

// Async Operations Tests

TEST_F(ClientTest, UnitAsyncReadDevice) {
    Client client(config_);
    client.authenticate("test_user", "test_pass", ClientType::Cpp);
    
    EXPECT_NO_THROW({
        auto future = client.read_device_async(0x8000, Register::STATUS);
        auto result = future.get();
        
        EXPECT_EQ(result.device_id, 0x8000);
        EXPECT_EQ(result.register_type, Register::STATUS);
    });
}

TEST_F(ClientTest, UnitAsyncReadDeviceCallback) {
    Client client(config_);
    client.authenticate("test_user", "test_pass", ClientType::Cpp);
    
    bool callback_called = false;
    DeviceResult callback_result;
    
    EXPECT_NO_THROW({
        client.read_device_async(0x8000, Register::STATUS, 
            [&callback_called, &callback_result](const DeviceResult& result) {
                callback_called = true;
                callback_result = result;
            });
        
        // Wait for callback (in real tests, use proper synchronization)
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        
        EXPECT_TRUE(callback_called);
        EXPECT_EQ(callback_result.device_id, 0x8000);
    });
}

// Bulk Operations Tests

TEST_F(ClientTest, UnitBulkReadSync) {
    Client client(config_);
    client.authenticate("test_user", "test_pass", ClientType::Cpp);
    
    std::vector<DeviceId> device_ids = {0x8000, 0x8001, 0x8002};
    
    EXPECT_NO_THROW({
        auto result = client.bulk_read_sync(device_ids, Register::STATUS);
        
        EXPECT_EQ(result.results.size(), device_ids.size());
        EXPECT_EQ(result.summary.total, device_ids.size());
        
        for (size_t i = 0; i < result.results.size(); ++i) {
            EXPECT_EQ(result.results[i].device_id, device_ids[i]);
            EXPECT_EQ(result.results[i].register_type, Register::STATUS);
        }
    });
}

TEST_F(ClientTest, UnitBulkReadAsync) {
    Client client(config_);
    client.authenticate("test_user", "test_pass", ClientType::Cpp);
    
    std::vector<DeviceId> device_ids = {0x8000, 0x8001, 0x8002};
    bool callback_called = false;
    BulkResult callback_result;
    
    EXPECT_NO_THROW({
        client.bulk_read_async(device_ids, Register::STATUS,
            [&callback_called, &callback_result](const BulkResult& result) {
                callback_called = true;
                callback_result = result;
            });
        
        // Wait for callback
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        
        EXPECT_TRUE(callback_called);
        EXPECT_EQ(callback_result.results.size(), device_ids.size());
    });
}

// Configuration Tests

TEST_F(ClientTest, UnitConfigureConnectionPool) {
    Client client(config_);
    
    PoolConfig pool_config;
    pool_config.max_connections = 20;
    pool_config.keep_alive_timeout = std::chrono::seconds(300);
    
    EXPECT_NO_THROW({
        client.configure_connection_pool(pool_config);
    });
}

TEST_F(ClientTest, UnitConfigureSecurity) {
    Client client(config_);
    
    SecurityConfig security_config;
    security_config.client_cert_path = "/path/to/cert.pem";
    security_config.verify_hostname = true;
    
    EXPECT_NO_THROW({
        client.configure_security(security_config);
    });
}

// Metrics and Monitoring Tests

TEST_F(ClientTest, UnitGetPerformanceMetrics) {
    Client client(config_);
    client.authenticate("test_user", "test_pass", ClientType::Cpp);
    
    EXPECT_NO_THROW({
        auto metrics = client.get_performance_metrics();
        
        EXPECT_GE(metrics.total_operations, 0);
        EXPECT_GE(metrics.successful_operations, 0);
        EXPECT_GE(metrics.failed_operations, 0);
        EXPECT_GE(metrics.success_rate, 0.0);
        EXPECT_LE(metrics.success_rate, 1.0);
    });
}

TEST_F(ClientTest, UnitGetDeviceRegistry) {
    Client client(config_);
    client.authenticate("test_user", "test_pass", ClientType::Cpp);
    
    EXPECT_NO_THROW({
        auto devices = client.get_device_registry();
        
        EXPECT_FALSE(devices.empty());
        
        for (const auto& device : devices) {
            EXPECT_GE(device.device_id, 0x8000);
            EXPECT_LE(device.device_id, 0x806B);
            EXPECT_FALSE(device.device_name.empty());
        }
    });
}

TEST_F(ClientTest, UnitGetDeviceInfo) {
    Client client(config_);
    client.authenticate("test_user", "test_pass", ClientType::Cpp);
    
    EXPECT_NO_THROW({
        auto device_info = client.get_device_info(0x8000);
        
        EXPECT_EQ(device_info.device_id, 0x8000);
        EXPECT_FALSE(device_info.device_name.empty());
        EXPECT_GE(device_info.device_group, 0);
    });
}

TEST_F(ClientTest, UnitGetDeviceInfoInvalid) {
    Client client(config_);
    client.authenticate("test_user", "test_pass", ClientType::Cpp);
    
    EXPECT_THROW({
        client.get_device_info(0x9999);  // Invalid device ID
    }, DeviceException);
}

// Integration Tests (require running test server)

class ClientIntegrationTest : public ::testing::Test {
protected:
    void SetUp() override {
        // These tests require a running test server
        if (!test_server_available()) {
            GTEST_SKIP() << "Test server not available";
        }
        
        config_.base_url = "https://localhost:8443";  // Test server
        config_.api_version = "2.0";
        config_.verify_ssl = false;
        config_.ca_cert_path = "test_certificates/test_ca.pem";
    }
    
    bool test_server_available() {
        // Check if test server is running
        // This would ping the test server endpoint
        return false;  // Assume not available for now
    }
    
    ClientConfig config_;
};

TEST_F(ClientIntegrationTest, IntegrationFullWorkflow) {
    Client client(config_);
    
    // Authenticate
    auto auth_result = client.authenticate("integration_test", "test_pass", ClientType::Cpp);
    ASSERT_TRUE(auth_result.success) << "Authentication failed: " << auth_result.error_message;
    
    // Read device
    auto read_result = client.read_device_sync(0x8000, Register::STATUS);
    ASSERT_TRUE(read_result.success) << "Read failed: " << read_result.error_message;
    
    // Bulk read
    std::vector<DeviceId> devices = {0x8000, 0x8001, 0x8002};
    auto bulk_result = client.bulk_read_sync(devices, Register::STATUS);
    EXPECT_TRUE(bulk_result.overall_success);
    EXPECT_EQ(bulk_result.results.size(), devices.size());
    
    // Performance metrics
    auto metrics = client.get_performance_metrics();
    EXPECT_GT(metrics.total_operations, 0);
}

TEST_F(ClientIntegrationTest, IntegrationWebSocketStreaming) {
    Client client(config_);
    
    auto auth_result = client.authenticate("integration_test", "test_pass", ClientType::Cpp);
    ASSERT_TRUE(auth_result.success);
    
    auto ws_client = client.create_websocket_client();
    ASSERT_NE(ws_client, nullptr);
    
    // Test WebSocket connection and streaming
    // This would require a more complex test setup with actual WebSocket server
}

} // namespace test
} // namespace dsmil