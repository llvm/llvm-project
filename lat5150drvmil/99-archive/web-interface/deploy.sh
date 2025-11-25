#!/bin/bash
# DSMIL Control System Deployment Script
# Phase 2 Track C - Web Interface Deployment

set -e

PROJECT_ROOT="/home/john/LAT5150DRVMIL/web-interface"
FRONTEND_DIR="$PROJECT_ROOT/frontend"
BACKEND_DIR="$PROJECT_ROOT/backend"
DATABASE_DIR="$PROJECT_ROOT/database"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
    exit 1
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."
    
    # Check for Node.js
    if ! command -v node &> /dev/null; then
        error "Node.js is required but not installed"
    fi
    
    # Check for Python 3.9+
    if ! command -v python3 &> /dev/null; then
        error "Python 3 is required but not installed"
    fi
    
    # Check for pip
    if ! command -v pip3 &> /dev/null; then
        error "pip3 is required but not installed"
    fi
    
    # Check for PostgreSQL (optional)
    if ! command -v psql &> /dev/null; then
        warning "PostgreSQL not found - will use SQLite fallback"
    fi
    
    # Check kernel module
    if [ ! -f "/home/john/LAT5150DRVMIL/01-source/kernel/dsmil-72dev.ko" ]; then
        warning "DSMIL kernel module not found - backend will run in simulation mode"
    fi
    
    success "Prerequisites check completed"
}

# Setup Python backend
setup_backend() {
    log "Setting up Python backend..."
    
    cd "$BACKEND_DIR"
    
    # Create virtual environment if it doesn't exist
    if [ ! -d "venv" ]; then
        log "Creating Python virtual environment..."
        python3 -m venv venv
    fi
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Install Python dependencies
    log "Installing Python dependencies..."
    pip install -r requirements.txt
    
    # Create .env file if it doesn't exist
    if [ ! -f ".env" ]; then
        log "Creating environment configuration..."
        cat > .env << EOF
DSMIL_DEBUG_MODE=true
DSMIL_SECRET_KEY=dsmil-dev-secret-key-$(openssl rand -hex 16)
DSMIL_DATABASE_URL=sqlite+aiosqlite:///./dsmil_control.db
DSMIL_CORS_ORIGINS=["http://localhost:3000","https://localhost:3000"]
DSMIL_LOG_LEVEL=INFO
EOF
    fi
    
    success "Backend setup completed"
}

# Setup React frontend
setup_frontend() {
    log "Setting up React frontend..."
    
    cd "$FRONTEND_DIR"
    
    # Install Node.js dependencies
    if [ ! -d "node_modules" ]; then
        log "Installing Node.js dependencies..."
        npm install
    else
        log "Updating Node.js dependencies..."
        npm update
    fi
    
    # Create public directory structure
    mkdir -p public
    
    # Create index.html if it doesn't exist
    if [ ! -f "public/index.html" ]; then
        log "Creating HTML template..."
        cat > public/index.html << EOF
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <meta name="theme-color" content="#000000" />
    <meta name="description" content="DSMIL Control System - Military-grade device management" />
    <title>DSMIL Control System</title>
  </head>
  <body>
    <noscript>You need to enable JavaScript to run this app.</noscript>
    <div id="root"></div>
  </body>
</html>
EOF
    fi
    
    success "Frontend setup completed"
}

# Setup database
setup_database() {
    log "Setting up database..."
    
    cd "$DATABASE_DIR"
    
    # For now, we'll use SQLite for development
    # PostgreSQL setup would be added here for production
    
    log "Database setup completed (using SQLite)"
}

# Load kernel module if available
load_kernel_module() {
    log "Checking kernel module..."
    
    KERNEL_MODULE="/home/john/LAT5150DRVMIL/01-source/kernel/dsmil-72dev.ko"
    
    if [ -f "$KERNEL_MODULE" ]; then
        if lsmod | grep -q "dsmil_72dev"; then
            log "DSMIL kernel module already loaded"
        else
            log "Loading DSMIL kernel module..."
            if sudo insmod "$KERNEL_MODULE" 2>/dev/null; then
                success "Kernel module loaded successfully"
            else
                warning "Failed to load kernel module - backend will run in simulation mode"
            fi
        fi
    else
        warning "Kernel module not found - backend will run in simulation mode"
    fi
}

# Start services
start_services() {
    log "Starting DSMIL Control System services..."
    
    # Start backend
    cd "$BACKEND_DIR"
    source venv/bin/activate
    
    log "Starting FastAPI backend on port 8000..."
    python main.py &
    BACKEND_PID=$!
    echo $BACKEND_PID > backend.pid
    
    sleep 3
    
    # Start frontend
    cd "$FRONTEND_DIR"
    log "Starting React frontend on port 3000..."
    npm start &
    FRONTEND_PID=$!
    echo $FRONTEND_PID > frontend.pid
    
    success "Services started successfully"
    log "Backend PID: $BACKEND_PID (saved to backend.pid)"
    log "Frontend PID: $FRONTEND_PID (saved to frontend.pid)"
    log ""
    log "DSMIL Control System is now running:"
    log "  Frontend: http://localhost:3000"
    log "  Backend API: http://localhost:8000"
    log "  API Documentation: http://localhost:8000/api/v1/docs"
    log ""
    log "Default login credentials:"
    log "  Admin: admin / dsmil_admin_2024"
    log "  Operator: operator / dsmil_op_2024"
    log "  Analyst: analyst / dsmil_analyst_2024"
}

# Stop services
stop_services() {
    log "Stopping DSMIL Control System services..."
    
    # Stop backend
    if [ -f "$BACKEND_DIR/backend.pid" ]; then
        BACKEND_PID=$(cat "$BACKEND_DIR/backend.pid")
        if kill -0 $BACKEND_PID 2>/dev/null; then
            kill $BACKEND_PID
            rm "$BACKEND_DIR/backend.pid"
            log "Backend stopped"
        fi
    fi
    
    # Stop frontend
    if [ -f "$FRONTEND_DIR/frontend.pid" ]; then
        FRONTEND_PID=$(cat "$FRONTEND_DIR/frontend.pid")
        if kill -0 $FRONTEND_PID 2>/dev/null; then
            kill $FRONTEND_PID
            rm "$FRONTEND_DIR/frontend.pid"
            log "Frontend stopped"
        fi
    fi
    
    success "All services stopped"
}

# Show status
show_status() {
    log "DSMIL Control System Status:"
    log ""
    
    # Check backend
    if [ -f "$BACKEND_DIR/backend.pid" ]; then
        BACKEND_PID=$(cat "$BACKEND_DIR/backend.pid")
        if kill -0 $BACKEND_PID 2>/dev/null; then
            success "Backend: Running (PID: $BACKEND_PID)"
        else
            error "Backend: Not running (stale PID file)"
        fi
    else
        warning "Backend: Not running"
    fi
    
    # Check frontend
    if [ -f "$FRONTEND_DIR/frontend.pid" ]; then
        FRONTEND_PID=$(cat "$FRONTEND_DIR/frontend.pid")
        if kill -0 $FRONTEND_PID 2>/dev/null; then
            success "Frontend: Running (PID: $FRONTEND_PID)"
        else
            error "Frontend: Not running (stale PID file)"
        fi
    else
        warning "Frontend: Not running"
    fi
    
    # Check kernel module
    if lsmod | grep -q "dsmil_72dev"; then
        success "Kernel Module: Loaded"
    else
        warning "Kernel Module: Not loaded (simulation mode)"
    fi
}

# Main deployment logic
case "${1:-deploy}" in
    "deploy")
        log "Starting DSMIL Control System deployment..."
        check_prerequisites
        setup_backend
        setup_frontend
        setup_database
        load_kernel_module
        start_services
        log "Deployment completed successfully!"
        ;;
    
    "start")
        load_kernel_module
        start_services
        ;;
    
    "stop")
        stop_services
        ;;
    
    "restart")
        stop_services
        sleep 2
        load_kernel_module
        start_services
        ;;
    
    "status")
        show_status
        ;;
    
    "setup")
        check_prerequisites
        setup_backend
        setup_frontend
        setup_database
        success "Setup completed - run './deploy.sh start' to launch services"
        ;;
    
    *)
        echo "Usage: $0 {deploy|start|stop|restart|status|setup}"
        echo ""
        echo "Commands:"
        echo "  deploy   - Full deployment (setup + start)"
        echo "  start    - Start services"
        echo "  stop     - Stop services"
        echo "  restart  - Restart services"
        echo "  status   - Show service status"
        echo "  setup    - Setup only (no start)"
        exit 1
        ;;
esac