# DSMIL Device Documentation

**Complete reference for all 80 implemented DSMIL devices**

## Summary Statistics

- **Total Devices:** 80
- **Total Operations:** 656
- **Total Registers:** 273
- **Average Operations per Device:** 8.2

## Most Complex Devices

- [0x8000: TPMControlDevice](0x8000.md) - 41 operations
- [0x8002: CredentialVaultDevice](0x8002.md) - 15 operations
- [0x8050: StorageEncryptionDevice](0x8050.md) - 14 operations
- [0x802A: NetworkMonitorDevice](0x802A.md) - 13 operations
- [0x8005: PerformanceMonitorDevice](0x8005.md) - 12 operations
- [0x8013: KeyManagementDevice](0x8013.md) - 12 operations
- [0x8017: RemoteAccessDevice](0x8017.md) - 12 operations
- [0x801E: TacticalDisplayDevice](0x801E.md) - 12 operations
- [0x805A: SensorArrayDevice](0x805A.md) - 12 operations
- [0x8001: BootSecurityDevice](0x8001.md) - 11 operations

## Devices by Group

### Group 0: Core Security (9 devices)

- 🟡 [0x8000: TPMControlDevice](0x8000.md) - 41 operations
- 🟡 [0x8001: BootSecurityDevice](0x8001.md) - 11 operations
- 🟡 [0x8002: CredentialVaultDevice](0x8002.md) - 15 operations
- 🟢 [0x8003: AuditLogDevice](0x8003.md) - 9 operations
- 🟢 [0x8004: EventLoggerDevice](0x8004.md) - 10 operations
- 🟢 [0x8005: PerformanceMonitorDevice](0x8005.md) - 12 operations
- 🟢 [0x8006: ThermalSensorDevice](0x8006.md) - 9 operations
- 🟡 [0x8007: PowerStateDevice](0x8007.md) - 9 operations
- 🟡 [0x8008: EmergencyResponseDevice](0x8008.md) - 11 operations

### Group 1: Extended Security (12 devices)

- 🟢 [0x800c: IntrusionDetectionDevice](0x800c.md) - 7 operations
- 🟢 [0x800d: BiometricAuthDevice](0x800d.md) - 7 operations
- 🟢 [0x800e: GeofenceControlDevice](0x800e.md) - 7 operations
- 🟢 [0x800f: KeyManagementDevice](0x800f.md) - 7 operations
- 🟢 [0x8010: IntrusionDetectionDevice](0x8010.md) - 6 operations
- 🟢 [0x8011: TokenManagerDevice](0x8011.md) - 7 operations
- 🟢 [0x8012: VPNControllerDevice](0x8012.md) - 7 operations
- 🟢 [0x8013: KeyManagementDevice](0x8013.md) - 12 operations
- 🟢 [0x8014: CertificateStoreDevice](0x8014.md) - 6 operations
- 🟢 [0x8015: RemoteDisableDevice](0x8015.md) - 7 operations
- 🟢 [0x8016: VPNControllerDevice](0x8016.md) - 9 operations
- 🟢 [0x8017: RemoteAccessDevice](0x8017.md) - 12 operations

### Group 2: Network/Communications (11 devices)

- 🟢 [0x8018: PreIsolationDevice](0x8018.md) - 11 operations
- 🟢 [0x801A: PortSecurityDevice](0x801A.md) - 5 operations
- 🟢 [0x801B: WirelessSecurityDevice](0x801B.md) - 5 operations
- 🟢 [0x801E: TacticalDisplayDevice](0x801E.md) - 12 operations
- 🟢 [0x801c: DataLinkDevice](0x801c.md) - 7 operations
- 🟢 [0x801d: SatelliteCommDevice](0x801d.md) - 7 operations
- 🟢 [0x801f: RadioControlDevice](0x801f.md) - 7 operations
- 🟢 [0x8020: FrequencyHopDevice](0x8020.md) - 7 operations
- 🟢 [0x8021: SystemResetDevice](0x8021.md) - 7 operations
- 🟢 [0x8022: NetworkMonitorDevice](0x8022.md) - 7 operations
- 🟢 [0x8023: PacketFilterDevice](0x8023.md) - 7 operations

### Group 3: Data Processing (11 devices)

- 🟢 [0x8024: DataProcessorDevice](0x8024.md) - 7 operations
- 🟢 [0x8025: CryptoAccelDevice](0x8025.md) - 7 operations
- 🟢 [0x8026: SignalAnalysisDevice](0x8026.md) - 7 operations
- 🟢 [0x8027: ImageProcessorDevice](0x8027.md) - 7 operations
- 🟢 [0x8028: VideoEncoderDevice](0x8028.md) - 7 operations
- 🟢 [0x802A: NetworkMonitorDevice](0x802A.md) - 13 operations
- 🟢 [0x802B: PacketFilterDevice](0x802B.md) - 6 operations
- 🟢 [0x802c: PatternRecognitionDevice](0x802c.md) - 7 operations
- 🟢 [0x802d: ThreatAnalysisDevice](0x802d.md) - 7 operations
- 🟢 [0x802e: TargetTrackingDevice](0x802e.md) - 7 operations
- 🟢 [0x802f: DataFusionDevice](0x802f.md) - 7 operations

### Group 4: Storage Management (12 devices)

- 🟢 [0x8030: StorageEncryptionDevice](0x8030.md) - 7 operations
- 🟢 [0x8031: SecureCacheDevice](0x8031.md) - 7 operations
- 🟢 [0x8032: RAIDControllerDevice](0x8032.md) - 7 operations
- 🟢 [0x8033: BackupManagerDevice](0x8033.md) - 7 operations
- 🟢 [0x8034: DataSanitizerDevice](0x8034.md) - 7 operations
- 🟢 [0x8035: StorageMonitorDevice](0x8035.md) - 7 operations
- 🟢 [0x8036: VolumeManagerDevice](0x8036.md) - 7 operations
- 🟢 [0x8037: SnapshotControlDevice](0x8037.md) - 7 operations
- 🟢 [0x8038: DeduplicationEngineDevice](0x8038.md) - 7 operations
- 🟢 [0x8039: CompressionEngineDevice](0x8039.md) - 7 operations
- 🟢 [0x803a: TieringControlDevice](0x803a.md) - 7 operations
- 🟢 [0x803b: CacheOptimizerDevice](0x803b.md) - 7 operations

### Group 5: Peripheral Control (12 devices)

- 🟢 [0x803c: SensorArrayDevice](0x803c.md) - 7 operations
- 🟢 [0x803d: ActuatorControlDevice](0x803d.md) - 7 operations
- 🟢 [0x803e: ServoManagerDevice](0x803e.md) - 7 operations
- 🟢 [0x803f: MotionControlDevice](0x803f.md) - 7 operations
- 🟢 [0x8040: HapticFeedbackDevice](0x8040.md) - 7 operations
- 🟢 [0x8041: DisplayControllerDevice](0x8041.md) - 7 operations
- 🟢 [0x8042: AudioOutputDevice](0x8042.md) - 7 operations
- 🟢 [0x8043: InputProcessorDevice](0x8043.md) - 7 operations
- 🟢 [0x8044: GestureRecognitionDevice](0x8044.md) - 7 operations
- 🟢 [0x8045: VoiceCommandDevice](0x8045.md) - 7 operations
- 🟢 [0x8046: BarcodeScannerDevice](0x8046.md) - 7 operations
- 🟢 [0x8047: RFIDReaderDevice](0x8047.md) - 7 operations

### Group 6: Training/Simulation (12 devices)

- 🟢 [0x8048: SimulationEngineDevice](0x8048.md) - 7 operations
- 🟢 [0x8049: ScenarioManagerDevice](0x8049.md) - 7 operations
- 🟢 [0x804a: TrainingRecorderDevice](0x804a.md) - 7 operations
- 🟢 [0x804b: PerformanceAnalyzerDevice](0x804b.md) - 7 operations
- 🟢 [0x804c: MissionPlannerDevice](0x804c.md) - 7 operations
- 🟢 [0x804d: TacticalOverlayDevice](0x804d.md) - 7 operations
- 🟢 [0x804e: DecisionSupportDevice](0x804e.md) - 7 operations
- 🟢 [0x804f: CollaborationHubDevice](0x804f.md) - 7 operations
- 🟢 [0x8050: StorageEncryptionDevice](0x8050.md) - 14 operations
- 🟢 [0x8051: ExpertSystemDevice](0x8051.md) - 7 operations
- 🟢 [0x8052: AdaptiveLearningDevice](0x8052.md) - 7 operations
- 🟢 [0x8053: AssessmentToolDevice](0x8053.md) - 7 operations

### Extended Range (1 devices)

- 🟢 [0x805A: SensorArrayDevice](0x805A.md) - 12 operations

## Quick Reference

### Finding a Device

Click on any device ID above to see its complete documentation.

### Risk Levels

- 🟢 **SAFE** - Standard operations, low risk
- 🟡 **MONITORED** - Security-critical, READ operations safe
- 🔴 **QUARANTINED** - Destructive operations, permanently blocked

### Common Operations

All devices support these core operations:
- `initialize()` - Initialize the device
- `get_status()` - Get current device status
- `get_capabilities()` - Get device capabilities

---

**Generated:** 2025-11-08  
**Framework Version:** 2.0.0  
**Total Coverage:** 80/108 devices (74.1%)  
**Classification:** UNCLASSIFIED // FOR OFFICIAL USE ONLY
