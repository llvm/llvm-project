# Secure DSMIL Phase 2 Deployment - Usage Guide

## Quick Start

### 1. Set Database Password (REQUIRED)
```bash
export DSMIL_DB_PASSWORD="your_actual_database_password"
```

### 2. Source Configuration (Optional)
```bash
source secure_deployment_config.env
```

### 3. Run Secure Deployment
```bash
python3 secure_accelerated_phase2_deployment.py
```

## Environment Variables

| Variable | Required | Default | Purpose |
|----------|----------|---------|---------|
| `DSMIL_DB_PASSWORD` | **YES** | None | Database password |
| `DSMIL_DB_HOST` | No | localhost | Database host |
| `DSMIL_DB_PORT` | No | 5433 | Database port |
| `DSMIL_USE_SUDO` | No | false | Enable sudo operations |
| `DSMIL_ENCRYPT_LOGS` | No | true | Encrypt sensitive logs |

## Security Features

- ✅ **Zero hardcoded passwords**
- ✅ **Environment-based configuration**  
- ✅ **Secure database connections**
- ✅ **Input validation and sanitization**
- ✅ **Command injection protection**
- ✅ **Secure file permissions**
- ✅ **Optional log encryption**
- ✅ **Proper error handling**

## Troubleshooting

### Database Connection Issues
```bash
# Check if PostgreSQL is running
docker ps | grep claude-postgres

# Test connection
psql -h localhost -p 5433 -U claude_agent -d claude_agents_auth
```

### Permission Issues
```bash
# Check required permissions
ls -la /var/log/dsmil/
ls -la /tmp/dsmil_secure/

# Fix if needed
sudo mkdir -p /var/log/dsmil
sudo chown $USER:$USER /var/log/dsmil
```

### TPM Issues
```bash
# Check TPM availability
tpm2_getcap properties-fixed 2>/dev/null | grep TPM_PT_FAMILY

# If not available, deployment will use simulation mode
```

## Security Best Practices

1. **Never commit passwords to version control**
2. **Use strong, unique database passwords**
3. **Regular security audits of deployment logs**
4. **Monitor file permissions on created files**
5. **Keep encryption keys secure**

---
*For full security details, see `SECURITY_FIXES_REPORT.md`*