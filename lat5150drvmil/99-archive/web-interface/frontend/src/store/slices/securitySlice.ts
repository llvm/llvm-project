import { createSlice, PayloadAction } from '@reduxjs/toolkit';
import { SecurityAlert, RiskLevel } from '../../types';

interface SecurityState {
  alerts: SecurityAlert[];
  threatLevel: RiskLevel;
  activeThreats: number;
  authFailures: number;
  systemHealth: {
    overallStatus: 'NORMAL' | 'WARNING' | 'CRITICAL' | 'EMERGENCY';
    cpuUsage: number;
    memoryUsage: number;
    diskUsage: number;
    temperature: number;
    networkConnectivity: boolean;
  };
  accessControl: {
    activeUsers: number;
    activeSessions: number;
    failedAttempts: number;
    lockedAccounts: string[];
  };
  auditMetrics: {
    eventsToday: number;
    criticalEvents: number;
    pendingInvestigations: number;
  };
  emergencyStop: {
    isActive: boolean;
    triggeredBy?: string;
    timestamp?: Date;
    reason?: string;
  };
  loading: boolean;
  error: string | null;
}

const initialState: SecurityState = {
  alerts: [],
  threatLevel: RiskLevel.LOW,
  activeThreats: 0,
  authFailures: 0,
  systemHealth: {
    overallStatus: 'NORMAL',
    cpuUsage: 0,
    memoryUsage: 0,
    diskUsage: 0,
    temperature: 0,
    networkConnectivity: true,
  },
  accessControl: {
    activeUsers: 0,
    activeSessions: 0,
    failedAttempts: 0,
    lockedAccounts: [],
  },
  auditMetrics: {
    eventsToday: 0,
    criticalEvents: 0,
    pendingInvestigations: 0,
  },
  emergencyStop: {
    isActive: false,
  },
  loading: false,
  error: null,
};

const securitySlice = createSlice({
  name: 'security',
  initialState,
  reducers: {
    // Alert management
    addAlert: (state, action: PayloadAction<SecurityAlert>) => {
      state.alerts.unshift(action.payload);
      
      // Update active threats count
      if (!action.payload.resolved) {
        state.activeThreats += 1;
      }
      
      // Update threat level based on severity
      if (action.payload.severity === 'CRITICAL' && state.threatLevel !== RiskLevel.CRITICAL) {
        state.threatLevel = RiskLevel.CRITICAL;
      } else if (action.payload.severity === 'HIGH' && state.threatLevel === RiskLevel.LOW) {
        state.threatLevel = RiskLevel.HIGH;
      }
      
      // Keep only last 500 alerts
      if (state.alerts.length > 500) {
        state.alerts = state.alerts.slice(0, 500);
      }
    },
    
    updateAlert: (state, action: PayloadAction<SecurityAlert>) => {
      const index = state.alerts.findIndex(alert => alert.id === action.payload.id);
      if (index !== -1) {
        const wasResolved = state.alerts[index].resolved;
        const isResolved = action.payload.resolved;
        
        state.alerts[index] = action.payload;
        
        // Update active threats count
        if (!wasResolved && isResolved) {
          state.activeThreats -= 1;
        } else if (wasResolved && !isResolved) {
          state.activeThreats += 1;
        }
      }
    },
    
    resolveAlert: (state, action: PayloadAction<string>) => {
      const alert = state.alerts.find(alert => alert.id === action.payload);
      if (alert && !alert.resolved) {
        alert.resolved = true;
        state.activeThreats -= 1;
      }
    },
    
    removeAlert: (state, action: PayloadAction<string>) => {
      const alert = state.alerts.find(alert => alert.id === action.payload);
      if (alert && !alert.resolved) {
        state.activeThreats -= 1;
      }
      state.alerts = state.alerts.filter(alert => alert.id !== action.payload);
    },
    
    clearResolvedAlerts: (state) => {
      state.alerts = state.alerts.filter(alert => !alert.resolved);
    },
    
    // Threat level management
    setThreatLevel: (state, action: PayloadAction<RiskLevel>) => {
      state.threatLevel = action.payload;
    },
    
    // System health updates
    updateSystemHealth: (state, action: PayloadAction<Partial<SecurityState['systemHealth']>>) => {
      state.systemHealth = { ...state.systemHealth, ...action.payload };
    },
    
    // Access control updates
    updateAccessControl: (state, action: PayloadAction<Partial<SecurityState['accessControl']>>) => {
      state.accessControl = { ...state.accessControl, ...action.payload };
    },
    
    addFailedAuth: (state) => {
      state.authFailures += 1;
      state.accessControl.failedAttempts += 1;
    },
    
    resetAuthFailures: (state) => {
      state.authFailures = 0;
    },
    
    lockAccount: (state, action: PayloadAction<string>) => {
      if (!state.accessControl.lockedAccounts.includes(action.payload)) {
        state.accessControl.lockedAccounts.push(action.payload);
      }
    },
    
    unlockAccount: (state, action: PayloadAction<string>) => {
      state.accessControl.lockedAccounts = state.accessControl.lockedAccounts.filter(
        account => account !== action.payload
      );
    },
    
    // Audit metrics
    updateAuditMetrics: (state, action: PayloadAction<Partial<SecurityState['auditMetrics']>>) => {
      state.auditMetrics = { ...state.auditMetrics, ...action.payload };
    },
    
    incrementAuditEvent: (state, action: PayloadAction<{ critical?: boolean }>) => {
      state.auditMetrics.eventsToday += 1;
      if (action.payload.critical) {
        state.auditMetrics.criticalEvents += 1;
      }
    },
    
    // Emergency stop management
    activateEmergencyStop: (state, action: PayloadAction<{ triggeredBy: string; reason: string }>) => {
      state.emergencyStop = {
        isActive: true,
        triggeredBy: action.payload.triggeredBy,
        timestamp: new Date(),
        reason: action.payload.reason,
      };
      
      // Set system to emergency status
      state.systemHealth.overallStatus = 'EMERGENCY';
      state.threatLevel = RiskLevel.CRITICAL;
    },
    
    deactivateEmergencyStop: (state) => {
      state.emergencyStop = {
        isActive: false,
      };
      
      // Reset system status (will be updated by real system status)
      state.systemHealth.overallStatus = 'NORMAL';
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
    
    // Reset security state
    resetSecurityState: (state) => {
      return { ...initialState };
    }
  },
});

export const {
  addAlert,
  updateAlert,
  resolveAlert,
  removeAlert,
  clearResolvedAlerts,
  setThreatLevel,
  updateSystemHealth,
  updateAccessControl,
  addFailedAuth,
  resetAuthFailures,
  lockAccount,
  unlockAccount,
  updateAuditMetrics,
  incrementAuditEvent,
  activateEmergencyStop,
  deactivateEmergencyStop,
  setLoading,
  setError,
  clearError,
  resetSecurityState
} = securitySlice.actions;

export default securitySlice.reducer;