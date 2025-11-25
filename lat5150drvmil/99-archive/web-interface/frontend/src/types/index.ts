// DSMIL Control System Type Definitions

export enum ClearanceLevel {
  UNCLASSIFIED = 0,
  RESTRICTED = 1,
  CONFIDENTIAL = 2,
  SECRET = 3,
  TOP_SECRET = 4,
  SCI = 5,
  SAP = 6,
  COSMIC = 7
}

export enum RiskLevel {
  SAFE = 'SAFE',
  LOW = 'LOW',
  MODERATE = 'MODERATE',
  HIGH = 'HIGH',
  CRITICAL = 'CRITICAL',
  QUARANTINED = 'QUARANTINED'
}

export enum DeviceStatus {
  UNKNOWN = 'UNKNOWN',
  ACTIVE = 'ACTIVE',
  INACTIVE = 'INACTIVE',
  ERROR = 'ERROR',
  QUARANTINED = 'QUARANTINED',
  MAINTENANCE = 'MAINTENANCE'
}

export enum OperationType {
  READ = 'READ',
  write = 'WRITE',
  config = 'CONFIG',
  reset = 'RESET',
  activate = 'ACTIVATE',
  deactivate = 'DEACTIVATE'
}

export interface UserContext {
  userId: string;
  username: string;
  clearanceLevel: ClearanceLevel;
  permissions: string[];
  sessionId: string;
  expiresAt: Date;
  compartmentAccess: string[];
  authorizedDevices: number[];
}

export interface DeviceInfo {
  deviceId: number;
  deviceName: string;
  deviceGroup: number;
  deviceIndex: number;
  status: DeviceStatus;
  riskLevel: RiskLevel;
  securityClassification: string;
  requiredClearance: ClearanceLevel;
  isQuarantined: boolean;
  lastAccessed?: Date;
  accessCount: number;
  errorCount: number;
  capabilities: Record<string, unknown>;
  constraints: Record<string, unknown>;
}

export interface DeviceOperation {
  operationId: string;
  deviceId: number;
  operationType: OperationType;
  operationData?: Record<string, unknown>;
  justification?: string;
  timestamp: Date;
  userId: string;
  result: 'SUCCESS' | 'DENIED' | 'ERROR' | 'EMERGENCY_STOP';
  executionTimeMs?: number;
  auditTrailId: string;
}

export interface SystemStatus {
  timestamp: Date;
  overallStatus: 'NORMAL' | 'WARNING' | 'CRITICAL' | 'EMERGENCY';
  deviceCount: number;
  activeDevices: number;
  quarantinedDevices: number[];
  systemHealth: {
    cpuUsage: number;
    memoryUsage: number;
    diskUsage: number;
    temperature: number;
  };
  securityStatus: {
    threatLevel: RiskLevel;
    activeAlerts: number;
    authFailures: number;
  };
  performanceMetrics: {
    operationsPerSecond: number;
    averageLatency: number;
    errorRate: number;
  };
}

export interface SecurityAlert {
  id: string;
  type: string;
  severity: 'LOW' | 'MODERATE' | 'HIGH' | 'CRITICAL';
  timestamp: Date;
  message: string;
  deviceId?: number;
  userId?: string;
  resolved: boolean;
  responseActions?: string[];
}

export interface AuditEvent {
  id: string;
  timestamp: Date;
  eventType: string;
  userId: string;
  deviceId?: number;
  operationType?: string;
  riskLevel: RiskLevel;
  result: string;
  details: Record<string, unknown>;
}

export interface WebSocketMessage {
  type: 'SYSTEM_STATUS_UPDATE' | 'SECURITY_ALERT' | 'DEVICE_STATE_CHANGED' | 'AUDIT_EVENT' | 'EMERGENCY_STOP';
  data: unknown;
  timestamp: Date;
}

// Application State Types
export interface DsmilApplicationState {
  // System state
  systemStatus: SystemStatus | null;
  deviceRegistry: DeviceInfo[];
  securityAlerts: SecurityAlert[];
  auditEvents: AuditEvent[];
  
  // User interface state
  currentUser: UserContext | null;
  activeView: string;
  selectedDevice: number | null;
  
  // Real-time data
  isConnected: boolean;
  lastUpdate: Date | null;
  
  // Loading and error states
  loading: boolean;
  error: string | null;
}

// Component Props
export interface SafeOperationProps {
  operation: {
    operationId: string;
    deviceId: number;
    operationType: OperationType;
    riskLevel: RiskLevel;
    requiredClearance: ClearanceLevel;
    requiresConfirmation: boolean;
    requiresDualAuth: boolean;
  };
  onExecute: (justification: string) => Promise<void>;
  onCancel: () => void;
}

export interface DeviceCardProps {
  device: DeviceInfo;
  onClick: (deviceId: number) => void;
  showControls?: boolean;
}

export interface SecurityBadgeProps {
  level: ClearanceLevel;
  size?: 'small' | 'medium' | 'large';
}

export interface RiskIndicatorProps {
  riskLevel: RiskLevel;
  showLabel?: boolean;
}