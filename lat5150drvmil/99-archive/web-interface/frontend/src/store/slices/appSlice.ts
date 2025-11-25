import { createSlice, PayloadAction } from '@reduxjs/toolkit';
import { DsmilApplicationState, UserContext, SystemStatus, SecurityAlert } from '../../types';

const initialState: DsmilApplicationState = {
  // System state
  systemStatus: null,
  deviceRegistry: [],
  securityAlerts: [],
  auditEvents: [],
  
  // User interface state
  currentUser: null,
  activeView: 'dashboard',
  selectedDevice: null,
  
  // Real-time data
  isConnected: false,
  lastUpdate: null,
  
  // Loading and error states
  loading: false,
  error: null,
};

const appSlice = createSlice({
  name: 'app',
  initialState,
  reducers: {
    // User authentication
    setUser: (state, action: PayloadAction<UserContext>) => {
      state.currentUser = action.payload;
    },
    
    clearUser: (state) => {
      state.currentUser = null;
      state.isConnected = false;
    },
    
    // System status
    setSystemStatus: (state, action: PayloadAction<SystemStatus>) => {
      state.systemStatus = action.payload;
      state.lastUpdate = new Date();
    },
    
    // Device registry
    setDeviceRegistry: (state, action: PayloadAction<any[]>) => {
      state.deviceRegistry = action.payload;
    },
    
    updateDeviceStatus: (state, action: PayloadAction<{ deviceId: number; status: any }>) => {
      const { deviceId, status } = action.payload;
      const deviceIndex = state.deviceRegistry.findIndex(d => d.deviceId === deviceId);
      if (deviceIndex !== -1) {
        state.deviceRegistry[deviceIndex] = { ...state.deviceRegistry[deviceIndex], ...status };
      }
    },
    
    // Security alerts
    addSecurityAlert: (state, action: PayloadAction<SecurityAlert>) => {
      state.securityAlerts.unshift(action.payload);
      // Keep only last 100 alerts
      if (state.securityAlerts.length > 100) {
        state.securityAlerts = state.securityAlerts.slice(0, 100);
      }
    },
    
    removeSecurityAlert: (state, action: PayloadAction<string>) => {
      state.securityAlerts = state.securityAlerts.filter(alert => alert.id !== action.payload);
    },
    
    markAlertResolved: (state, action: PayloadAction<string>) => {
      const alert = state.securityAlerts.find(alert => alert.id === action.payload);
      if (alert) {
        alert.resolved = true;
      }
    },
    
    // UI state
    setActiveView: (state, action: PayloadAction<string>) => {
      state.activeView = action.payload;
    },
    
    setSelectedDevice: (state, action: PayloadAction<number | null>) => {
      state.selectedDevice = action.payload;
    },
    
    // Connection state
    setConnectionStatus: (state, action: PayloadAction<boolean>) => {
      state.isConnected = action.payload;
      if (action.payload) {
        state.lastUpdate = new Date();
      }
    },
    
    // Loading and error states
    setLoading: (state, action: PayloadAction<boolean>) => {
      state.loading = action.payload;
    },
    
    setError: (state, action: PayloadAction<string | null>) => {
      state.error = action.payload;
    },
    
    clearError: (state) => {
      state.error = null;
    },
    
    // Update last activity
    updateLastActivity: (state) => {
      state.lastUpdate = new Date();
    }
  },
});

export const {
  setUser,
  clearUser,
  setSystemStatus,
  setDeviceRegistry,
  updateDeviceStatus,
  addSecurityAlert,
  removeSecurityAlert,
  markAlertResolved,
  setActiveView,
  setSelectedDevice,
  setConnectionStatus,
  setLoading,
  setError,
  clearError,
  updateLastActivity
} = appSlice.actions;

export default appSlice.reducer;