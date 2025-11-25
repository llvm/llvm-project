# 🖥️ DSMIL Control System - Track C Web Interface

**Classification**: RESTRICTED  
**Version**: 1.0.0  
**Date**: 2025-09-01  

Military-grade web-based control panel for 84 DSMIL devices with real-time monitoring, comprehensive security, and audit logging.

## 🏗️ System Architecture

### Frontend (React + TypeScript)
- **Military-themed UI** with dark mode and security indicators
- **Real-time updates** via WebSocket connections  
- **Safety-first interface** with quarantine warnings and risk indicators
- **Multi-level authentication** with clearance-based access control
- **Responsive design** for desktop and mobile access

### Backend (FastAPI + Python)
- **RESTful API** with OpenAPI documentation
- **WebSocket support** for real-time device updates
- **Multi-factor authentication** with JWT tokens
- **Comprehensive audit logging** with integrity protection
- **Risk-based authorization** for device operations

### Database (PostgreSQL + SQLite)
- **PostgreSQL** for production deployment
- **SQLite** fallback for development
- **Comprehensive audit trails** with chain integrity
- **Performance metrics** tracking
- **User session management**

### Integration Layer
- **Kernel Module Interface** - Direct integration with Track A (dsmil-72dev.ko)
- **Device Controller** - Hardware abstraction layer with safety checks
- **Security Monitor** - Real-time threat detection and system health
- **Emergency Stop** - Fail-safe system shutdown capabilities

## 📋 Key Features

### 🛡️ Security Features
- **Multi-level clearance** system (RESTRICTED → TOP_SECRET)
- **Risk-based operations** with automatic threat assessment
- **Quarantine management** for critical devices (5 quarantined devices)
- **Real-time security monitoring** with automated threat detection
- **Comprehensive audit logging** with cryptographic integrity
- **Emergency stop** capability for immediate system shutdown

### 🎯 Device Management
- **84 DSMIL devices** organized in 7 groups (12 devices each)
- **Real-time status monitoring** for all devices
- **Safe device operations** (READ, WRITE, CONFIG, RESET)
- **Quarantine protection** with automatic safety checks
- **Performance metrics** tracking per device
- **Historical operation logging** with full audit trails

### 📊 Monitoring & Analytics
- **System health dashboard** with CPU, memory, disk, temperature
- **Performance metrics** - operations/sec, latency, error rates
- **Security alerts** with severity-based notifications
- **Device status visualization** with risk level indicators
- **Real-time updates** via WebSocket connections
- **Comprehensive reporting** for compliance and analysis

## 🚀 Quick Start

### Prerequisites
- **Python 3.9+** with pip
- **Node.js 16+** with npm
- **PostgreSQL 13+** (optional, SQLite fallback available)
- **DSMIL Kernel Module** (dsmil-72dev.ko) for hardware integration

### 1. Clone and Deploy
```bash
cd /home/john/LAT5150DRVMIL/web-interface
./deploy.sh deploy
```

### 2. Access the System
- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/api/v1/docs

### 3. Default Credentials
| User | Password | Clearance | Access |
|------|----------|-----------|---------|
| admin | dsmil_admin_2024 | TOP_SECRET | All 84 devices |
| operator | dsmil_op_2024 | SECRET | First 60 devices |
| analyst | dsmil_analyst_2024 | CONFIDENTIAL | First 36 devices |

## 📖 Detailed Documentation

### System Components

#### Frontend Structure
```
frontend/src/
├── components/          # Reusable UI components
├── pages/              # Page components (Dashboard, Devices, etc.)
├── store/              # Redux state management
├── services/           # API and WebSocket services
├── theme/              # Military UI theme configuration
├── types/              # TypeScript type definitions
└── utils/              # Helper utilities
```

#### Backend Structure  
```
backend/
├── main.py             # FastAPI application entry point
├── config.py           # Application configuration
├── auth.py             # Authentication & authorization
├── device_controller.py # Device hardware interface
├── websocket_manager.py # Real-time communication
├── audit_logger.py     # Comprehensive audit logging
├── security_monitor.py # Threat detection & system health
└── models.py           # Database models
```

### Device Organization

**84 DSMIL Devices** organized as follows:
- **Group 0** (0x8000-0x800B): Master Security Controllers (CRITICAL risk)
- **Group 1** (0x800C-0x8017): Primary System Interfaces (HIGH risk)
- **Group 2** (0x8018-0x8023): Secondary System Interfaces (HIGH risk)
- **Group 3** (0x8024-0x802F): Thermal Management (MODERATE risk)
- **Group 4** (0x8030-0x803B): Power Management (MODERATE risk)
- **Group 5** (0x803C-0x8047): Auxiliary Control (LOW risk)
- **Group 6** (0x8048-0x8053): **5 QUARANTINED DEVICES** (QUARANTINED status)

### API Endpoints

#### System Status
- `GET /api/v1/system/status` - Get comprehensive system status
- `GET /api/v1/health` - Basic health check

#### Device Management
- `GET /api/v1/devices` - List accessible devices
- `POST /api/v1/devices/{device_id}/operations` - Execute device operation
- `GET /api/v1/devices/{device_id}/statistics` - Get device statistics

#### Security & Emergency
- `POST /api/v1/emergency-stop` - Trigger emergency stop
- `GET /api/v1/security/alerts` - Get security alerts
- `POST /api/v1/security/alerts/{alert_id}/resolve` - Resolve security alert

#### WebSocket Endpoints
- `WS /api/v1/ws` - Real-time updates with authentication

### Safety & Security

#### Risk Assessment Matrix
| Operation | Risk Level | Required Clearance | Dual Auth |
|-----------|------------|-------------------|-----------|
| READ (LOW device) | LOW | RESTRICTED | No |
| READ (CRITICAL device) | HIGH | SECRET | No |
| WRITE (any device) | HIGH | SECRET | Recommended |
| CONFIG/RESET | CRITICAL | TOP_SECRET | Required |

#### Quarantine Protection
- **5 quarantined devices** (0x8048-0x804C) require TOP_SECRET clearance
- **Automatic safety checks** before any operation
- **Emergency stop** capability accessible to all authenticated users
- **Real-time monitoring** with automated threat detection

## 🔧 Development & Operations

### Development Mode
```bash
# Backend development with hot reload
cd backend
source venv/bin/activate
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Frontend development with hot reload  
cd frontend
npm start
```

### Production Deployment
```bash
# Build production frontend
cd frontend
npm run build

# Run backend in production mode
cd backend
export DSMIL_DEBUG_MODE=false
python main.py
```

### Service Management
```bash
./deploy.sh start     # Start services
./deploy.sh stop      # Stop services
./deploy.sh restart   # Restart services
./deploy.sh status    # Check status
./deploy.sh setup     # Setup only (no start)
```

### Monitoring & Logs
- **Backend logs**: Structured logging with security event tracking
- **Frontend errors**: Browser console and error boundaries
- **System metrics**: Real-time monitoring via WebSocket
- **Audit trail**: Complete operation history with integrity protection

## 🔍 Troubleshooting

### Common Issues

#### Kernel Module Not Loading
```bash
# Check if module exists
ls -la /home/john/LAT5150DRVMIL/01-source/kernel/dsmil-72dev.ko

# Load module manually
sudo insmod /home/john/LAT5150DRVMIL/01-source/kernel/dsmil-72dev.ko

# Check if loaded
lsmod | grep dsmil
```

#### Backend Connection Issues
```bash
# Check backend process
ps aux | grep python | grep main.py

# Check port availability
netstat -tulpn | grep :8000

# Check logs
tail -f backend/logs/dsmil.log
```

#### Frontend Build Issues
```bash
# Clear node modules and reinstall
cd frontend
rm -rf node_modules package-lock.json
npm install

# Check for TypeScript errors
npm run lint
```

### Performance Optimization

#### Backend Performance
- **Database connection pooling** (20 connections, 30 max overflow)
- **Async request handling** with FastAPI
- **WebSocket connection management** with automatic cleanup
- **Caching** for frequently accessed device information

#### Frontend Performance  
- **Component memoization** with React.memo
- **State management** optimized with Redux Toolkit
- **Lazy loading** for large device lists
- **WebSocket connection sharing** across components

## 📝 Change Log

### Version 1.0.0 (2025-09-01)
- ✅ Initial implementation of Track C web interface
- ✅ Military-themed React frontend with TypeScript
- ✅ FastAPI backend with comprehensive security
- ✅ PostgreSQL database integration with audit logging
- ✅ Real-time WebSocket communication
- ✅ Integration with Track A kernel module (dsmil-72dev)
- ✅ Safety-first device control with quarantine protection
- ✅ Multi-level authentication and authorization
- ✅ Emergency stop functionality
- ✅ Comprehensive system monitoring and alerting

## 🤝 Integration with Other Tracks

### Track A (Kernel Module) Integration
- **Direct IOCTL interface** to dsmil-72dev kernel module
- **Hardware abstraction layer** with safety checks
- **Simulation mode** when kernel module unavailable
- **Performance monitoring** of kernel operations

### Track B (Security Layer) Integration  
- **Multi-level clearance** system implementation
- **Risk-based authorization** for device operations
- **Comprehensive audit logging** with integrity protection
- **Real-time security monitoring** and threat detection

### Future Enhancements
- **Advanced analytics** with machine learning threat detection
- **Mobile app** for remote monitoring
- **Integration APIs** for external security systems
- **Advanced reporting** with compliance frameworks

---

**Classification**: RESTRICTED  
**Document Control**: DSMIL-TRACK-C-001  
**Last Updated**: 2025-09-01 by Web Agent Team

For technical support or security concerns, contact the DSMIL development team.