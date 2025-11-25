import React, { useEffect, useState } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { ThemeProvider } from '@mui/material/styles';
import { CssBaseline, Box } from '@mui/material';
import { Provider } from 'react-redux';

import { theme } from './theme/militaryTheme';
import { store } from './store';
import { useAppDispatch, useAppSelector } from './store/hooks';

// Components
import NavigationHeader from './components/NavigationHeader';
import SecurityStatusBar from './components/SecurityStatusBar';
import NotificationSystem from './components/NotificationSystem';
import EmergencyStopButton from './components/EmergencyStopButton';
import LoadingOverlay from './components/LoadingOverlay';

// Pages
import DashboardPage from './pages/DashboardPage';
import DeviceManagementPage from './pages/DeviceManagementPage';
import SecurityMonitoringPage from './pages/SecurityMonitoringPage';
import AuditLogPage from './pages/AuditLogPage';
import OperationsConsolePage from './pages/OperationsConsolePage';
import EmergencyControlPage from './pages/EmergencyControlPage';
import LoginPage from './pages/LoginPage';

// Services
import { websocketService } from './services/websocketService';
import { authService } from './services/authService';

// Store actions
import { setUser, setSystemStatus, addSecurityAlert } from './store/slices/appSlice';

// Types
import { UserContext, WebSocketMessage } from './types';

const AppContent: React.FC = () => {
  const dispatch = useAppDispatch();
  const { currentUser, loading, error } = useAppSelector(state => state.app);
  const [isInitialized, setIsInitialized] = useState(false);

  useEffect(() => {
    const initializeApp = async () => {
      try {
        // Check for existing authentication
        const token = localStorage.getItem('dsmil_token');
        if (token) {
          const user = await authService.verifyToken(token);
          if (user) {
            dispatch(setUser(user));
            
            // Initialize WebSocket connection
            await websocketService.connect(token);
            
            // Set up WebSocket message handlers
            websocketService.onMessage((message: WebSocketMessage) => {
              handleWebSocketMessage(message);
            });
          }
        }
      } catch (error) {
        console.error('Failed to initialize app:', error);
        localStorage.removeItem('dsmil_token');
      } finally {
        setIsInitialized(true);
      }
    };

    initializeApp();

    // Cleanup on unmount
    return () => {
      websocketService.disconnect();
    };
  }, [dispatch]);

  const handleWebSocketMessage = (message: WebSocketMessage) => {
    switch (message.type) {
      case 'SYSTEM_STATUS_UPDATE':
        dispatch(setSystemStatus(message.data));
        break;
        
      case 'SECURITY_ALERT':
        dispatch(addSecurityAlert(message.data));
        break;
        
      case 'DEVICE_STATE_CHANGED':
        // Handle device state updates
        console.log('Device state changed:', message.data);
        break;
        
      case 'AUDIT_EVENT':
        // Handle audit events
        console.log('Audit event:', message.data);
        break;
        
      case 'EMERGENCY_STOP':
        // Handle emergency stop
        console.warn('Emergency stop activated:', message.data);
        break;
        
      default:
        console.warn('Unknown WebSocket message type:', message.type);
    }
  };

  // Show loading screen during initialization
  if (!isInitialized) {
    return <LoadingOverlay message="Initializing DSMIL Control System..." />;
  }

  // Show login page if not authenticated
  if (!currentUser) {
    return <LoginPage />;
  }

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', minHeight: '100vh' }}>
      {/* Navigation Header */}
      <NavigationHeader />
      
      {/* Security Status Bar */}
      <SecurityStatusBar />
      
      {/* Main Content Area */}
      <Box component="main" sx={{ flexGrow: 1, p: 3 }}>
        <Routes>
          <Route path="/" element={<DashboardPage />} />
          <Route path="/dashboard" element={<Navigate to="/" replace />} />
          <Route path="/devices" element={<DeviceManagementPage />} />
          <Route path="/security" element={<SecurityMonitoringPage />} />
          <Route path="/audit" element={<AuditLogPage />} />
          <Route path="/operations" element={<OperationsConsolePage />} />
          <Route path="/emergency" element={<EmergencyControlPage />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </Box>
      
      {/* Notification System */}
      <NotificationSystem />
      
      {/* Emergency Stop Button */}
      <EmergencyStopButton />
      
      {/* Loading Overlay */}
      {loading && <LoadingOverlay message="Processing..." />}
      
      {/* Error Display */}
      {error && (
        <Box
          sx={{
            position: 'fixed',
            bottom: 16,
            right: 16,
            backgroundColor: 'error.main',
            color: 'error.contrastText',
            padding: 2,
            borderRadius: 1,
            maxWidth: 400,
            zIndex: 9999
          }}
        >
          {error}
        </Box>
      )}
    </Box>
  );
};

const App: React.FC = () => {
  return (
    <Provider store={store}>
      <ThemeProvider theme={theme}>
        <CssBaseline />
        <Router>
          <AppContent />
        </Router>
      </ThemeProvider>
    </Provider>
  );
};

export default App;