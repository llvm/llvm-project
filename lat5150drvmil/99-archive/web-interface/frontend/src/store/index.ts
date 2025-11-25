import { configureStore } from '@reduxjs/toolkit';
import appSlice from './slices/appSlice';
import devicesSlice from './slices/devicesSlice';
import securitySlice from './slices/securitySlice';
import auditSlice from './slices/auditSlice';

export const store = configureStore({
  reducer: {
    app: appSlice,
    devices: devicesSlice,
    security: securitySlice,
    audit: auditSlice
  },
  middleware: (getDefaultMiddleware) =>
    getDefaultMiddleware({
      serializableCheck: {
        // Ignore these action types
        ignoredActions: ['app/setSystemStatus', 'app/addSecurityAlert'],
        // Ignore these field paths in all actions
        ignoredActionsPaths: ['payload.timestamp', 'payload.expiresAt'],
        // Ignore these paths in the state
        ignoredPaths: ['app.systemStatus.timestamp', 'app.currentUser.expiresAt'],
      },
    }),
});

export type RootState = ReturnType<typeof store.getState>;
export type AppDispatch = typeof store.dispatch;