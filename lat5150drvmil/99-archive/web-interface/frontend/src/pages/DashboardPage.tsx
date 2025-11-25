import React, { useState, useEffect } from 'react';
import {
  Box,
  Grid,
  Card,
  CardHeader,
  CardContent,
  Typography,
  Alert,
  Chip,
  LinearProgress,
  IconButton,
  Button,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Tooltip
} from '@mui/material';
import {
  Dashboard as DashboardIcon,
  Security as SecurityIcon,
  Warning as WarningIcon,
  CheckCircle as CheckCircleIcon,
  Error as ErrorIcon,
  DeviceHub as DeviceHubIcon,
  Speed as SpeedIcon,
  Thermostat as ThermostatIcon,
  Memory as MemoryIcon,
  Storage as StorageIcon,
  NetworkCheck as NetworkIcon,
  Emergency as EmergencyIcon,
  Refresh as RefreshIcon,
  Info as InfoIcon
} from '@mui/icons-material';

import { useAppSelector, useAppDispatch } from '../store/hooks';
import { getRiskLevelColor, getClearanceLevelColor } from '../theme/militaryTheme';
import { SystemStatus, DeviceInfo, SecurityAlert, RiskLevel } from '../types';

const DashboardPage: React.FC = () => {
  const dispatch = useAppDispatch();
  const { systemStatus, deviceRegistry, securityAlerts, currentUser } = useAppSelector(state => state.app);
  const [selectedAlert, setSelectedAlert] = useState<SecurityAlert | null>(null);
  const [detailsDialogOpen, setDetailsDialogOpen] = useState(false);

  // Real-time data updates would be handled by WebSocket in production
  useEffect(() => {
    // Simulate periodic updates
    const interval = setInterval(() => {
      // Updates would come via WebSocket
    }, 30000);

    return () => clearInterval(interval);
  }, []);

  const handleAlertClick = (alert: SecurityAlert) => {
    setSelectedAlert(alert);
    setDetailsDialogOpen(true);
  };

  const handleRefresh = () => {
    // Trigger system status refresh
    window.location.reload(); // Simple refresh for demo
  };

  const getSystemStatusColor = (status: string) => {
    switch (status) {
      case 'NORMAL': return '#4CAF50';
      case 'WARNING': return '#FF9800';
      case 'CRITICAL': return '#F44336';
      case 'EMERGENCY': return '#9C27B0';
      default: return '#9E9E9E';
    }
  };

  const getDeviceStatusIcon = (status: string, isQuarantined: boolean) => {
    if (isQuarantined) return <ErrorIcon color="error" />;
    
    switch (status) {
      case 'ACTIVE': return <CheckCircleIcon color="success" />;
      case 'INACTIVE': return <WarningIcon color="warning" />;
      case 'ERROR': return <ErrorIcon color="error" />;
      default: return <InfoIcon color="info" />;
    }
  };

  const systemHealth = systemStatus?.system_health || {
    cpuUsage: 0,
    memoryUsage: 0,
    diskUsage: 0,
    temperature: 0
  };

  const activeDevices = deviceRegistry?.filter(d => d.status === 'ACTIVE').length || 0;
  const quarantinedDevices = deviceRegistry?.filter(d => d.isQuarantined).length || 0;
  const totalDevices = deviceRegistry?.length || 84;

  const recentAlerts = securityAlerts?.slice(0, 5) || [];
  const criticalAlerts = securityAlerts?.filter(a => a.severity === 'CRITICAL' && !a.resolved) || [];

  return (
    <Box sx={{ p: 3 }}>
      {/* Page Header */}
      <Box sx={{ mb: 3, display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <Typography variant="h4" component="h1" sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
          <DashboardIcon sx={{ fontSize: '2rem' }} />
          DSMIL Control Dashboard
        </Typography>
        
        <Box sx={{ display: 'flex', gap: 2, alignItems: 'center' }}>
          <Chip 
            label={`Clearance: ${currentUser?.clearanceLevel || 'UNKNOWN'}`}
            sx={{ 
              backgroundColor: getClearanceLevelColor(currentUser?.clearanceLevel || 'UNKNOWN'),
              color: 'white',
              fontWeight: 'bold'
            }}
          />
          <Tooltip title="Refresh System Status">
            <IconButton onClick={handleRefresh}>
              <RefreshIcon />
            </IconButton>
          </Tooltip>
        </Box>
      </Box>

      {/* Critical Alerts */}
      {criticalAlerts.length > 0 && (
        <Alert 
          severity="error" 
          sx={{ mb: 3 }}
          action={
            <Button color="inherit" size="small" onClick={() => handleAlertClick(criticalAlerts[0])}>
              VIEW DETAILS
            </Button>
          }
        >
          {criticalAlerts.length} Critical Security Alert{criticalAlerts.length !== 1 ? 's' : ''} Require Immediate Attention
        </Alert>
      )}

      <Grid container spacing={3}>
        {/* System Status Overview */}
        <Grid item xs={12} md={6} lg={3}>
          <Card>
            <CardHeader 
              title="System Status"
              avatar={<SecurityIcon />}
              sx={{ backgroundColor: getSystemStatusColor(systemStatus?.overallStatus || 'UNKNOWN') + '20' }}
            />
            <CardContent>
              <Box sx={{ textAlign: 'center' }}>
                <Typography variant="h4" sx={{ color: getSystemStatusColor(systemStatus?.overallStatus || 'UNKNOWN') }}>
                  {systemStatus?.overallStatus || 'UNKNOWN'}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Last Updated: {systemStatus?.timestamp ? new Date(systemStatus.timestamp).toLocaleTimeString() : 'Unknown'}
                </Typography>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        {/* Device Status */}
        <Grid item xs={12} md={6} lg={3}>
          <Card>
            <CardHeader 
              title="Device Status"
              avatar={<DeviceHubIcon />}
            />
            <CardContent>
              <Box sx={{ mb: 2 }}>
                <Typography variant="body2" color="text.secondary">Active Devices</Typography>
                <Typography variant="h5">{activeDevices} / {totalDevices}</Typography>
              </Box>
              
              <Box sx={{ mb: 2 }}>
                <Typography variant="body2" color="text.secondary">Quarantined</Typography>
                <Typography variant="h6" color="error.main">{quarantinedDevices}</Typography>
              </Box>
              
              <LinearProgress 
                variant="determinate" 
                value={(activeDevices / totalDevices) * 100}
                sx={{ height: 8, borderRadius: 4 }}
              />
            </CardContent>
          </Card>
        </Grid>

        {/* Security Alerts */}
        <Grid item xs={12} md={6} lg={3}>
          <Card>
            <CardHeader 
              title="Security Alerts"
              avatar={<WarningIcon />}
            />
            <CardContent>
              <Box sx={{ mb: 2 }}>
                <Typography variant="body2" color="text.secondary">Active Alerts</Typography>
                <Typography variant="h5" color={criticalAlerts.length > 0 ? 'error.main' : 'success.main'}>
                  {securityAlerts?.filter(a => !a.resolved).length || 0}
                </Typography>
              </Box>
              
              <Box sx={{ mb: 2 }}>
                <Typography variant="body2" color="text.secondary">Critical</Typography>
                <Typography variant="h6" color="error.main">{criticalAlerts.length}</Typography>
              </Box>
              
              <Button 
                variant="outlined" 
                size="small" 
                fullWidth
                onClick={() => window.location.href = '/security'}
              >
                View All Alerts
              </Button>
            </CardContent>
          </Card>
        </Grid>

        {/* Performance Metrics */}
        <Grid item xs={12} md={6} lg={3}>
          <Card>
            <CardHeader 
              title="Performance"
              avatar={<SpeedIcon />}
            />
            <CardContent>
              <Box sx={{ mb: 1 }}>
                <Typography variant="body2" color="text.secondary">Operations/sec</Typography>
                <Typography variant="h6">
                  {systemStatus?.performance_metrics?.operationsPerSecond || 0}
                </Typography>
              </Box>
              
              <Box sx={{ mb: 1 }}>
                <Typography variant="body2" color="text.secondary">Avg Latency</Typography>
                <Typography variant="body1">
                  {systemStatus?.performance_metrics?.averageLatency || 0}ms
                </Typography>
              </Box>
              
              <Box>
                <Typography variant="body2" color="text.secondary">Error Rate</Typography>
                <Typography variant="body1" color={systemStatus?.performance_metrics?.errorRate > 5 ? 'error.main' : 'success.main'}>
                  {systemStatus?.performance_metrics?.errorRate || 0}%
                </Typography>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        {/* System Health Metrics */}
        <Grid item xs={12} lg={6}>
          <Card>
            <CardHeader title="System Health" />
            <CardContent>
              <Grid container spacing={2}>
                {/* CPU Usage */}
                <Grid item xs={6}>
                  <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                    <SpeedIcon sx={{ mr: 1, color: 'primary.main' }} />
                    <Typography variant="body2">CPU Usage</Typography>
                  </Box>
                  <LinearProgress 
                    variant="determinate" 
                    value={systemHealth.cpuUsage}
                    color={systemHealth.cpuUsage > 80 ? 'error' : systemHealth.cpuUsage > 60 ? 'warning' : 'success'}
                    sx={{ height: 8, borderRadius: 4 }}
                  />
                  <Typography variant="caption" color="text.secondary">
                    {systemHealth.cpuUsage.toFixed(1)}%
                  </Typography>
                </Grid>

                {/* Memory Usage */}
                <Grid item xs={6}>
                  <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                    <MemoryIcon sx={{ mr: 1, color: 'primary.main' }} />
                    <Typography variant="body2">Memory</Typography>
                  </Box>
                  <LinearProgress 
                    variant="determinate" 
                    value={systemHealth.memoryUsage}
                    color={systemHealth.memoryUsage > 85 ? 'error' : systemHealth.memoryUsage > 70 ? 'warning' : 'success'}
                    sx={{ height: 8, borderRadius: 4 }}
                  />
                  <Typography variant="caption" color="text.secondary">
                    {systemHealth.memoryUsage.toFixed(1)}%
                  </Typography>
                </Grid>

                {/* Disk Usage */}
                <Grid item xs={6}>
                  <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                    <StorageIcon sx={{ mr: 1, color: 'primary.main' }} />
                    <Typography variant="body2">Disk Usage</Typography>
                  </Box>
                  <LinearProgress 
                    variant="determinate" 
                    value={systemHealth.diskUsage}
                    color={systemHealth.diskUsage > 90 ? 'error' : systemHealth.diskUsage > 75 ? 'warning' : 'success'}
                    sx={{ height: 8, borderRadius: 4 }}
                  />
                  <Typography variant="caption" color="text.secondary">
                    {systemHealth.diskUsage.toFixed(1)}%
                  </Typography>
                </Grid>

                {/* Temperature */}
                <Grid item xs={6}>
                  <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                    <ThermostatIcon sx={{ mr: 1, color: 'primary.main' }} />
                    <Typography variant="body2">Temperature</Typography>
                  </Box>
                  <LinearProgress 
                    variant="determinate" 
                    value={(systemHealth.temperature / 100) * 100}
                    color={systemHealth.temperature > 85 ? 'error' : systemHealth.temperature > 70 ? 'warning' : 'success'}
                    sx={{ height: 8, borderRadius: 4 }}
                  />
                  <Typography variant="caption" color="text.secondary">
                    {systemHealth.temperature.toFixed(1)}°C
                  </Typography>
                </Grid>
              </Grid>
            </CardContent>
          </Card>
        </Grid>

        {/* Recent Security Alerts */}
        <Grid item xs={12} lg={6}>
          <Card>
            <CardHeader 
              title="Recent Security Alerts" 
              action={
                <Button 
                  size="small" 
                  onClick={() => window.location.href = '/security'}
                >
                  View All
                </Button>
              }
            />
            <CardContent>
              {recentAlerts.length === 0 ? (
                <Typography color="text.secondary" sx={{ textAlign: 'center', py: 2 }}>
                  No recent security alerts
                </Typography>
              ) : (
                <List>
                  {recentAlerts.map((alert) => (
                    <ListItem 
                      key={alert.id}
                      button
                      onClick={() => handleAlertClick(alert)}
                      sx={{ 
                        borderLeft: `4px solid ${getRiskLevelColor(alert.severity)}`,
                        mb: 1,
                        borderRadius: 1,
                        backgroundColor: 'background.paper'
                      }}
                    >
                      <ListItemIcon>
                        {alert.severity === 'CRITICAL' ? (
                          <ErrorIcon color="error" />
                        ) : alert.severity === 'HIGH' ? (
                          <WarningIcon color="warning" />
                        ) : (
                          <InfoIcon color="info" />
                        )}
                      </ListItemIcon>
                      <ListItemText 
                        primary={alert.message}
                        secondary={
                          <Box>
                            <Typography variant="caption">
                              {new Date(alert.timestamp).toLocaleString()}
                            </Typography>
                            {alert.deviceId && (
                              <Chip 
                                size="small" 
                                label={`Device: ${alert.deviceId.toString(16).toUpperCase()}`}
                                sx={{ ml: 1 }}
                              />
                            )}
                            <Chip 
                              size="small" 
                              label={alert.severity}
                              sx={{ 
                                ml: 1,
                                backgroundColor: getRiskLevelColor(alert.severity),
                                color: 'white'
                              }}
                            />
                          </Box>
                        }
                      />
                    </ListItem>
                  ))}
                </List>
              )}
            </CardContent>
          </Card>
        </Grid>

        {/* Device Overview */}
        <Grid item xs={12}>
          <Card>
            <CardHeader title="Device Overview" />
            <CardContent>
              <Grid container spacing={2}>
                {deviceRegistry?.slice(0, 12).map((device) => (
                  <Grid item xs={12} sm={6} md={4} lg={2} key={device.deviceId}>
                    <Card 
                      variant="outlined" 
                      sx={{ 
                        cursor: 'pointer',
                        transition: 'all 0.3s ease',
                        '&:hover': {
                          boxShadow: 2,
                          borderColor: 'primary.main'
                        },
                        borderColor: device.isQuarantined ? 'error.main' : 'divider'
                      }}
                      onClick={() => window.location.href = `/devices`}
                    >
                      <CardContent sx={{ p: 2, '&:last-child': { pb: 2 } }}>
                        <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                          {getDeviceStatusIcon(device.status, device.isQuarantined)}
                          <Typography variant="caption" sx={{ ml: 1 }}>
                            {device.deviceName}
                          </Typography>
                        </Box>
                        
                        <Typography variant="body2" sx={{ fontFamily: 'monospace' }}>
                          ID: {device.deviceId.toString(16).toUpperCase()}
                        </Typography>
                        
                        <Chip 
                          size="small"
                          label={device.riskLevel}
                          sx={{
                            mt: 1,
                            backgroundColor: getRiskLevelColor(device.riskLevel),
                            color: 'white',
                            fontSize: '0.7rem'
                          }}
                        />
                      </CardContent>
                    </Card>
                  </Grid>
                ))}
              </Grid>
              
              {deviceRegistry && deviceRegistry.length > 12 && (
                <Box sx={{ textAlign: 'center', mt: 2 }}>
                  <Button 
                    variant="outlined" 
                    onClick={() => window.location.href = '/devices'}
                  >
                    View All {deviceRegistry.length} Devices
                  </Button>
                </Box>
              )}
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Alert Details Dialog */}
      <Dialog 
        open={detailsDialogOpen} 
        onClose={() => setDetailsDialogOpen(false)}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle>
          Security Alert Details
          {selectedAlert && (
            <Chip 
              size="small"
              label={selectedAlert.severity}
              sx={{ 
                ml: 2,
                backgroundColor: getRiskLevelColor(selectedAlert.severity),
                color: 'white'
              }}
            />
          )}
        </DialogTitle>
        <DialogContent>
          {selectedAlert && (
            <Box>
              <Typography variant="h6" gutterBottom>
                {selectedAlert.message}
              </Typography>
              
              <Typography variant="body2" color="text.secondary" gutterBottom>
                Detected: {new Date(selectedAlert.timestamp).toLocaleString()}
              </Typography>
              
              {selectedAlert.deviceId && (
                <Typography variant="body2" gutterBottom>
                  Affected Device: {selectedAlert.deviceId.toString(16).toUpperCase()}
                </Typography>
              )}
              
              {selectedAlert.userId && (
                <Typography variant="body2" gutterBottom>
                  Related User: {selectedAlert.userId}
                </Typography>
              )}
              
              {selectedAlert.responseActions && selectedAlert.responseActions.length > 0 && (
                <Box sx={{ mt: 2 }}>
                  <Typography variant="subtitle2" gutterBottom>
                    Recommended Actions:
                  </Typography>
                  <List>
                    {selectedAlert.responseActions.map((action, index) => (
                      <ListItem key={index}>
                        <ListItemText primary={action} />
                      </ListItem>
                    ))}
                  </List>
                </Box>
              )}
            </Box>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setDetailsDialogOpen(false)}>
            Close
          </Button>
          {selectedAlert && !selectedAlert.resolved && (
            <Button variant="contained" color="primary">
              Mark as Resolved
            </Button>
          )}
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default DashboardPage;