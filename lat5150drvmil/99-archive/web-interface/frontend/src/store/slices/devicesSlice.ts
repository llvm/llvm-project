import { createSlice, PayloadAction } from '@reduxjs/toolkit';
import { DeviceInfo, DeviceOperation, OperationType } from '../../types';

interface DevicesState {
  devices: DeviceInfo[];
  selectedDevice: DeviceInfo | null;
  activeOperations: DeviceOperation[];
  deviceStatistics: Record<number, any>;
  loading: boolean;
  error: string | null;
}

const initialState: DevicesState = {
  devices: [],
  selectedDevice: null,
  activeOperations: [],
  deviceStatistics: {},
  loading: false,
  error: null,
};

const devicesSlice = createSlice({
  name: 'devices',
  initialState,
  reducers: {
    // Device management
    setDevices: (state, action: PayloadAction<DeviceInfo[]>) => {
      state.devices = action.payload;
    },
    
    addDevice: (state, action: PayloadAction<DeviceInfo>) => {
      state.devices.push(action.payload);
    },
    
    updateDevice: (state, action: PayloadAction<DeviceInfo>) => {
      const index = state.devices.findIndex(d => d.deviceId === action.payload.deviceId);
      if (index !== -1) {
        state.devices[index] = action.payload;
        
        // Update selected device if it's the same one
        if (state.selectedDevice?.deviceId === action.payload.deviceId) {
          state.selectedDevice = action.payload;
        }
      }
    },
    
    removeDevice: (state, action: PayloadAction<number>) => {
      state.devices = state.devices.filter(d => d.deviceId !== action.payload);
      
      // Clear selection if removed device was selected
      if (state.selectedDevice?.deviceId === action.payload) {
        state.selectedDevice = null;
      }
    },
    
    // Device selection
    selectDevice: (state, action: PayloadAction<number>) => {
      const device = state.devices.find(d => d.deviceId === action.payload);
      state.selectedDevice = device || null;
    },
    
    clearDeviceSelection: (state) => {
      state.selectedDevice = null;
    },
    
    // Device operations
    addOperation: (state, action: PayloadAction<DeviceOperation>) => {
      state.activeOperations.push(action.payload);
    },
    
    updateOperation: (state, action: PayloadAction<DeviceOperation>) => {
      const index = state.activeOperations.findIndex(op => op.operationId === action.payload.operationId);
      if (index !== -1) {
        state.activeOperations[index] = action.payload;
      }
    },
    
    removeOperation: (state, action: PayloadAction<string>) => {
      state.activeOperations = state.activeOperations.filter(op => op.operationId !== action.payload);
    },
    
    clearCompletedOperations: (state) => {
      state.activeOperations = state.activeOperations.filter(
        op => op.result === 'SUCCESS' || op.result === 'DENIED' || op.result === 'ERROR' || op.result === 'EMERGENCY_STOP'
      );
    },
    
    // Device statistics
    setDeviceStatistics: (state, action: PayloadAction<{ deviceId: number; statistics: any }>) => {
      const { deviceId, statistics } = action.payload;
      state.deviceStatistics[deviceId] = statistics;
    },
    
    clearDeviceStatistics: (state, action: PayloadAction<number>) => {
      delete state.deviceStatistics[action.payload];
    },
    
    // Device status updates
    updateDeviceStatus: (state, action: PayloadAction<{ deviceId: number; status: any }>) => {
      const { deviceId, status } = action.payload;
      const device = state.devices.find(d => d.deviceId === deviceId);
      if (device) {
        Object.assign(device, status);
        
        // Update selected device if it's the same one
        if (state.selectedDevice?.deviceId === deviceId) {
          Object.assign(state.selectedDevice, status);
        }
      }
    },
    
    // Quarantine operations
    quarantineDevice: (state, action: PayloadAction<number>) => {
      const device = state.devices.find(d => d.deviceId === action.payload);
      if (device) {
        device.isQuarantined = true;
        device.status = 'QUARANTINED' as any;
        
        if (state.selectedDevice?.deviceId === action.payload) {
          state.selectedDevice.isQuarantined = true;
          state.selectedDevice.status = 'QUARANTINED' as any;
        }
      }
    },
    
    releaseFromQuarantine: (state, action: PayloadAction<number>) => {
      const device = state.devices.find(d => d.deviceId === action.payload);
      if (device) {
        device.isQuarantined = false;
        device.status = 'INACTIVE' as any; // Reset to inactive, will be updated by system
        
        if (state.selectedDevice?.deviceId === action.payload) {
          state.selectedDevice.isQuarantined = false;
          state.selectedDevice.status = 'INACTIVE' as any;
        }
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
    }
  },
});

export const {
  setDevices,
  addDevice,
  updateDevice,
  removeDevice,
  selectDevice,
  clearDeviceSelection,
  addOperation,
  updateOperation,
  removeOperation,
  clearCompletedOperations,
  setDeviceStatistics,
  clearDeviceStatistics,
  updateDeviceStatus,
  quarantineDevice,
  releaseFromQuarantine,
  setLoading,
  setError,
  clearError
} = devicesSlice.actions;

export default devicesSlice.reducer;